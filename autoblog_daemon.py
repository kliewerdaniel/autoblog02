#!/usr/bin/env python3
"""
Autoblog Continuous Generation Daemon

Runs the blog generation pipeline continuously, triggered by RSS feed updates.
Supports both manual one-off mode and continuous daemon mode.

Usage:
    # Manual mode (one-off generation)
    python3 autoblog_daemon.py --mode manual

    # Continuous daemon mode
    python3 autoblog_daemon.py --mode daemon

    # Or use environment variable
    export AUTOBLOG_MODE=daemon && python3 autoblog_daemon.py
"""

import asyncio
import sys
import os
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
import aiohttp
import feedparser
import yaml
import re
import json
import subprocess
import frontmatter
from slugify import slugify

# Add paths for agent imports
current_dir = Path(__file__).parent
agent_path = current_dir / "agent"
if str(agent_path) not in sys.path:
    sys.path.insert(0, str(agent_path))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from agent.orchestrator import BlogGenerationOrchestrator
from agent.llm_client import OllamaClient
from agent.config import config
from agent.vector_store import vector_store
from agent.models import DocumentChunk
from agent.utils.parser import chunk_content, clean_markdown
from sentence_transformers import SentenceTransformer

# Scheduling
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import threading
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoblog_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArticleData:
    """Data class for fetched RSS articles with deduplication GUID."""
    def __init__(self, title: str, content: str, url: str, source: str, published: datetime, guid: str = None):
        self.title = title
        self.content = content
        self.url = url
        self.source = source
        self.published = published
        self.guid = guid or url  # Use URL as GUID if not provided


class ProcessedItemsTracker:
    """Tracks processed RSS items to avoid duplicates."""

    def __init__(self, storage_file: str = "processed_items.json"):
        self.storage_file = Path(storage_file)
        self.processed = self._load_processed()

    def _load_processed(self) -> Dict[str, Dict]:
        """Load previously processed items."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    return data.get('processed_items', {})
            except Exception as e:
                logger.warning(f"Failed to load processed items: {e}")
        return {}

    def _save_processed(self):
        """Save processed items to disk."""
        data = {
            'processed_items': self.processed,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save processed items: {e}")

    def is_processed(self, guid: str) -> bool:
        """Check if item was already processed."""
        return guid in self.processed

    def mark_processed(self, guid: str, metadata: Dict = None):
        """Mark item as processed."""
        self.processed[guid] = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_processed()

    def get_last_processed_timestamp(self, feed_url: str) -> Optional[datetime]:
        """Get the timestamp of the last processed item for a feed."""
        # Find the most recent timestamp for this feed
        feed_items = [(guid, data) for guid, data in self.processed.items()
                     if data.get('metadata', {}).get('feed_url') == feed_url]

        if not feed_items:
            return None

        most_recent = max(feed_items, key=lambda x: x[1]['timestamp'])
        return datetime.fromisoformat(most_recent[1]['timestamp'])


class RSSIngestor:
    """Handles ingestion of RSS articles into the vector database."""

    def __init__(self):
        self.logger = logger
        self.embed_model = SentenceTransformer(config.embedding_model)

    async def ingest_articles(self, articles: List[ArticleData]) -> Dict[str, Any]:
        """Ingest new RSS articles into the vector database."""
        self.logger.info(f"Ingesting {len(articles)} new RSS articles into knowledge base...")

        if not articles:
            return {"error": "No new articles to ingest"}

        processed_texts = []
        processed_metadata = []
        processed_ids = []
        total_chunks = 0

        for i, article in enumerate(articles):
            try:
                # Clean content for better embeddings
                clean_content = clean_markdown(article.content)

                # Chunk the content
                chunks = chunk_content(
                    clean_content,
                    chunk_size=config.chunk_size,
                    overlap=config.chunk_overlap
                )

                if not chunks:
                    self.logger.warning(f"No chunks generated for article: {article.title}")
                    continue

                # Generate embeddings for this article's chunks
                embeddings = self.embed_model.encode(chunks, show_progress_bar=False)

                # Process each chunk
                for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    metadata = {
                        "source_type": "rss_feed",
                        "source_file": f"rss_{i}_{article.source.replace(' ', '_')}",
                        "title": article.title,
                        "url": article.url,
                        "source": article.source,
                        "date": article.published.isoformat() if article.published else None,
                        "guid": article.guid,
                        "categories": "News, Current Events",
                        "tags": f"rss, {article.source.replace(' ', '').lower()}, news",
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "excerpt": article.content[:200] + "..." if len(article.content) > 200 else article.content,
                        "word_count": len(article.content.split()),
                        "ingested_at": datetime.now().isoformat(),
                    }

                    processed_texts.append(chunk)
                    processed_metadata.append(metadata)
                    processed_ids.append(f"rss_{article.guid}_chunk_{j}")

                total_chunks += len(chunks)
                self.logger.info(f"Processed article {i+1}/{len(articles)}: {article.title} ({len(chunks)} chunks)")

            except Exception as e:
                self.logger.error(f"Failed to process article {article.title}: {e}")
                continue

        # Store all chunks in vector database if there are any
        if processed_texts:
            try:
                self.logger.info(f"Storing {len(processed_texts)} chunks in vector database...")

                # Generate embeddings for all texts at once
                batch_embeddings = self.embed_model.encode(processed_texts, show_progress_bar=False)

                vector_store.add_documents(
                    texts=processed_texts,
                    embeddings=batch_embeddings,
                    metadata=processed_metadata,
                    ids=processed_ids
                )

                self.logger.info(f"Successfully stored {len(processed_texts)} chunks in vector database")
                self.logger.info(f"Successfully ingested {total_chunks} chunks from {len(articles)} RSS articles into knowledge base")

                return {
                    "success": True,
                    "articles_ingested": len(articles),
                    "chunks_created": total_chunks,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Failed to store RSS articles: {e}")
                return {"error": f"Storage failed: {e}"}
        else:
            return {"error": "No chunks were processed"}


class FeedFetcher:
    """Enhanced RSS fetcher with incremental support."""

    def __init__(self, feeds_file: str = "feeds.yaml"):
        self.feeds_file = feeds_file
        self.logger = logger
        self.max_articles_per_feed = 50  # Increased for incremental
        self.min_article_length = 100
        self.tracker = ProcessedItemsTracker()

    async def fetch_new_feeds(self) -> List[ArticleData]:
        """Fetch only new articles from RSS feeds since last run."""
        self.logger.info("Fetching new RSS feed items...")

        with open(self.feeds_file, 'r') as f:
            feeds_config = yaml.safe_load(f)

        feeds = feeds_config.get('feeds', [])
        new_articles = []

        async with aiohttp.ClientSession() as session:
            for feed_url in feeds:
                feed_url = feed_url.strip()
                try:
                    articles = await self.fetch_single_feed_incremental(session, feed_url)
                    new_articles.extend(articles)
                except Exception as e:
                    self.logger.error(f"Failed to fetch from {feed_url}: {e}")

        self.logger.info(f"Found {len(new_articles)} new articles across all feeds")
        return new_articles

    async def fetch_single_feed_incremental(self, session: aiohttp.ClientSession, feed_url: str) -> List[ArticleData]:
        """Fetch new articles from a single feed, skipping already processed ones."""
        try:
            async with session.get(feed_url, timeout=30) as response:
                response.raise_for_status()
                content = await response.text()

            feed = feedparser.parse(content)
            new_articles = []

            last_processed = self.tracker.get_last_processed_timestamp(feed_url)
            self.logger.debug(f"Last processed timestamp for {feed_url}: {last_processed}")

            for entry in feed.entries:
                guid = entry.get('id', entry.get('link', ''))
                if not guid:
                    continue

                if self.tracker.is_processed(guid):
                    self.logger.debug(f"Skipping already processed: {guid}")
                    continue

                # Check timestamp if we have a last processed time
                publish_time = self.parse_date(entry)
                if last_processed and publish_time and publish_time <= last_processed:
                    self.logger.debug(f"Skipping older article: {publish_time} <= {last_processed}")
                    continue

                article_content = self.extract_content(entry)
                if len(article_content) < self.min_article_length:
                    continue

                article = ArticleData(
                    title=entry.get('title', 'No Title'),
                    content=article_content,
                    url=entry.get('link', ''),
                    source=feed.feed.get('title', feed_url),
                    published=publish_time,
                    guid=guid
                )

                new_articles.append(article)

                # Mark as processed immediately to avoid duplicates
                self.tracker.mark_processed(guid, {'feed_url': feed_url, 'title': article.title})

                if len(new_articles) >= self.max_articles_per_feed:
                    break

            self.logger.info(f"Found {len(new_articles)} new articles from {feed_url}")
            return new_articles

        except Exception as e:
            self.logger.warning(f"Error fetching {feed_url}: {e}")
            return []

    def extract_content(self, entry) -> str:
        """Extract and clean content from RSS entry."""
        content = ""
        for field in ['content', 'summary', 'description']:
            if hasattr(entry, field):
                if field == 'content' and entry.content:
                    content = entry.content[0].value if entry.content else ""
                else:
                    content = getattr(entry, field, "")
                break

        return re.sub(r'<[^>]+>', '', content).strip()

    def parse_date(self, entry) -> datetime:
        """Parse publication date from RSS entry."""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6])
        except:
            pass
        return datetime.now()


class TaskQueue:
    """Simple in-process task queue for the daemon."""

    def __init__(self):
        self.tasks = []
        self.lock = threading.Lock()
        self.logger = logger

    def enqueue(self, task_type: str, data: Dict):
        """Add a task to the queue."""
        with self.lock:
            task = {
                'id': f"{task_type}_{datetime.now().isoformat()}",
                'type': task_type,
                'data': data,
                'created_at': datetime.now().isoformat(),
                'status': 'pending'
            }
            self.tasks.append(task)
            self.logger.info(f"Enqueued task: {task['id']}")

    def dequeue(self) -> Optional[Dict]:
        """Get next pending task."""
        with self.lock:
            for task in self.tasks:
                if task['status'] == 'pending':
                    task['status'] = 'processing'
                    return task
            return None

    def complete_task(self, task_id: str):
        """Mark task as completed."""
        with self.lock:
            for task in self.tasks:
                if task['id'] == task_id:
                    task['status'] = 'completed'
                    task['completed_at'] = datetime.now().isoformat()
                    self.logger.info(f"Completed task: {task_id}")
                    break


class PublicationService:
    """Handles publishing blog posts to the content directory."""

    def __init__(self, content_dir: str = "content/blog"):
        self.content_dir = Path(content_dir)
        self.logger = logger
        self.content_dir.mkdir(parents=True, exist_ok=True)

    def publish_post(self, title: str, content: str, metadata: Dict = None) -> Dict[str, Any]:
        """Publish a blog post with frontmatter."""
        try:
            self.logger.info(f"Publishing blog post: {title}")

            # Generate slug for filename
            slug = slugify(title, max_length=100, word_boundary=True, save_order=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_str}-{slug}.md"
            filepath = self.content_dir / filename

            # Ensure unique filename
            counter = 1
            while filepath.exists():
                filename = f"{date_str}-{slug}-{counter}.md"
                filepath = self.content_dir / filename
                counter += 1

            # Prepare frontmatter
            default_metadata = {
                'title': title,
                'date': datetime.now().isoformat(),
                'categories': ['News', 'Current Events'],
                'tags': ['automated', 'news', 'analysis'],
                'draft': False,
                'excerpt': content[:200] + "..." if len(content) > 200 else content,
                'word_count': len(content.split()),
                'generated_by': 'autoblog-daemon'
            }

            if metadata:
                default_metadata.update(metadata)

            # Create post with frontmatter
            post = frontmatter.Post(content, **default_metadata)

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(frontmatter.dumps(post))

            self.logger.info(f"Published post to: {filepath}")

            # Stub: Trigger Next.js rebuild
            self._trigger_nextjs_rebuild()

            return {
                "success": True,
                "filepath": str(filepath),
                "slug": slug,
                "title": title
            }

        except Exception as e:
            self.logger.error(f"Failed to publish post: {e}")
            return {"error": str(e)}

    def _trigger_nextjs_rebuild(self):
        """Stub for triggering Next.js rebuild."""
        self.logger.info("Stub: Would trigger Next.js rebuild / revalidation here")
        # TODO: Implement webhook call, npm run build, or file system trigger
        # For now, just log the intent
        try:
            # Example: subprocess.run(['npm', 'run', 'build'], cwd='.')
            # Or webhook to Netlify/Vercel
            self.logger.info("Next.js rebuild notification stubbed - implement webhook or build trigger")
        except Exception as e:
            self.logger.warning(f"Rebuild trigger failed (stubbed): {e}")


class BlogGenerationService:
    """Service for generating blog posts from RSS data."""

    def __init__(self):
        self.orchestrator = BlogGenerationOrchestrator()
        self.topic_generator = BlogTopicGenerator()
        self.publication_service = PublicationService()
        self.logger = logger

    async def generate_from_feed_items(self, articles: List[ArticleData], dry_run: bool = False, has_new_articles: bool = True) -> Dict[str, Any]:
        """Generate blog post from RSS feed items."""
        if not articles and has_new_articles:
            return {"error": "No articles provided"}

        if not articles and not has_new_articles:
            self.logger.info("Generating blog post from existing knowledge")
        else:
            self.logger.info(f"Generating blog post from {len(articles)} RSS items")

        try:
            # Create topic from articles or generate original content
            if not has_new_articles:
                topic_prompt = await self.topic_generator.generate_original_topic()
            else:
                topic_prompt = await self.topic_generator.summarize_to_topic(articles)

            spec_data = {
                'topic': topic_prompt,
                'style': 'technical',
                'length': 'long',
                'tone': 'informative',
                'min_words': 1500,
                'max_words': 5000,
                'categories': ['News', 'Analysis', 'Current Events'],
                'tags': ['news', 'trends', 'analysis', 'rss', 'synthesis'] if has_new_articles else ['original', 'analysis', 'deep-dive', 'thought-leadership'],
                'has_new_articles': has_new_articles
            }

            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "topic": topic_prompt,
                    "would_generate": True
                }

            result = await self.orchestrator.generate_blog_post(topic_prompt, spec_data)

            if result.success and result.file_path:
                # The agent ingestor already saves the file to content/blog, which is the publication location
                self.logger.info(f"Blog post already published by agent workflow: {result.file_path}")

                return {
                    "success": True,
                    "file_path": result.file_path,
                    "published_path": result.file_path,  # Same location
                    "iterations": result.iterations,
                    "published": True  # Already published by agent
                }
            else:
                return {
                    "success": False,
                    "error": result.error,
                    "iterations": 0
                }

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {"error": str(e)}


class BlogTopicGenerator:
    """Generates initial blog topics from RSS content."""

    def __init__(self):
        self.ollama_client = OllamaClient()
        self.logger = logger

    async def summarize_to_topic(self, articles: List[ArticleData]) -> str:
        """Create a blog topic prompt from RSS articles."""
        if not articles:
            return "News Summary and Current Events Analysis"

        articles_text = []
        for i, article in enumerate(articles[:20]):
            articles_text.append(f"Article {i+1}:\nTitle: {article.title}\nContent: {article.content[:500]}...\nSource: {article.source}\n")

        articles_summary = "\n".join(articles_text)

        prompt = f"""
Based on the following collection of recent news articles, please:

1. Summarize the key themes and topics covered
2. Identify any connecting patterns or trends across the stories
3. Suggest a compelling blog post topic that weaves together these themes
4. Create a detailed writing prompt for an AI blog generator

Articles:
{articles_summary}

Please respond with:
- A brief summary of the articles
- The suggested blog post topic
- A detailed writing prompt suitable for an AI blog generator
"""

        self.logger.info("Generating initial blog topic from RSS content...")

        try:
            result = await self.ollama_client.generate(
                prompt,
                system_prompt="You are a creative blog topic strategist who finds connections in news and creates compelling writing prompts.",
                temperature=0.7
            )
            return result if result else "News Summary and Blog Topic"
        except Exception as e:
            self.logger.error(f"Error generating topic: {e}")
            return "News Summary and Blog Topic"

    async def generate_original_topic(self) -> str:
        """Generate an original blog topic when no new articles are available."""
        prompt = """You are a blog strategist tasked with creating compelling and valuable content based on existing knowledge in a news/analysis blog. Since no new articles are available right now, please:

1. Reflect on trending topics and evolving narratives in current events, technology, culture, or society
2. Identify a specific angle or deep-dive topic that would be engaging for readers
3. Create an original, insightful perspective that offers value through analysis or foresight
4. Suggest a detailed topic that could be expanded into a comprehensive blog post

Please provide:
- The suggested original blog post topic
- A brief rationale for why this topic would be valuable
- A detailed writing prompt for the AI blog generator, focusing on creating insightful analysis or forward-looking perspectives

Topic should be suitable for a 1500-5000 word blog post with technical depth and informative tone."""

        self.logger.info("Generating original blog topic from existing knowledge...")

        try:
            result = await self.ollama_client.generate(
                prompt,
                system_prompt="You are a creative strategist who generates original, valuable blog content based on patterns in existing knowledge and current trends.",
                temperature=0.8
            )
            return result if result else "Emerging Trends and Future Insights"
        except Exception as e:
            self.logger.error(f"Error generating original topic: {e}")
            return "Emerging Trends and Future Insights"


class ContinuousDaemon:
    """Main daemon class for continuous blog generation."""

    def __init__(self, check_interval_minutes: int = 30):
        self.check_interval = check_interval_minutes
        self.feed_fetcher = FeedFetcher()
        self.rss_ingestor = RSSIngestor()
        self.task_queue = TaskQueue()
        self.blog_service = BlogGenerationService()
        self.logger = logger
        self.running = True
        self.scheduler = None

    def start(self):
        """Start the continuous daemon."""
        self.logger.info(f"Starting autoblog daemon with {self.check_interval} minute intervals")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start the task processor in a background thread
        processor_thread = threading.Thread(target=self._task_processor_loop, daemon=True)
        processor_thread.start()

        # Set up scheduler
        self.scheduler = BlockingScheduler()
        self.scheduler.add_job(
            self._check_feeds_job,
            trigger=IntervalTrigger(minutes=self.check_interval),
            id='feed_check',
            name='Check RSS feeds for new content',
            max_instances=1
        )

        # Perform initial check immediately
        self.logger.info("Performing initial RSS feed check...")
        self._check_feeds_job()

        self.logger.info("Daemon started. Press Ctrl+C to stop.")

        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            self.logger.info("Daemon interrupted, shutting down...")
            self.stop()

    def stop(self):
        """Stop the daemon gracefully."""
        self.logger.info("Stopping autoblog daemon...")
        self.running = False
        if self.scheduler:
            self.scheduler.shutdown()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()

    def _check_feeds_job(self):
        """Scheduled job to check for new RSS content."""
        try:
            self.logger.info("Starting scheduled RSS feed check cycle...")
            asyncio.run(self._check_feeds_async())
            self.logger.info("Completed scheduled RSS feed check cycle - daemon will check again in next interval")
        except Exception as e:
            self.logger.error(f"Feed check job failed: {e}")

    async def _check_feeds_async(self):
        """Async version of feed checking."""
        self.logger.info("Checking for new RSS feed items...")

        try:
            new_articles = await self.feed_fetcher.fetch_new_feeds()

            if new_articles:
                self.logger.info(f"Found {len(new_articles)} new articles, enqueueing generation task")

                # Ingest the articles first
                ingestion_result = await self.rss_ingestor.ingest_articles(new_articles)

                if 'error' in ingestion_result:
                    self.logger.error(f"Ingestion failed: {ingestion_result['error']}")
                    return

                # Enqueue generation task with new articles
                self.task_queue.enqueue('generate_from_feed', {
                    'articles': [article.__dict__ for article in new_articles],
                    'ingestion_result': ingestion_result,
                    'has_new_articles': True
                })
            else:
                self.logger.info("No new articles found - generating post from existing knowledge")
                # Enqueue generation task with empty articles (will generate from existing knowledge)
                self.task_queue.enqueue('generate_from_feed', {
                    'articles': [],
                    'ingestion_result': None,
                    'has_new_articles': False
                })

        except Exception as e:
            self.logger.error(f"Feed check failed: {e}")

    def _task_processor_loop(self):
        """Background loop to process queued tasks."""
        while self.running:
            try:
                task = self.task_queue.dequeue()
                if task:
                    asyncio.run(self._process_task(task))
                else:
                    time.sleep(1)  # Wait before checking queue again
            except Exception as e:
                self.logger.error(f"Task processor error: {e}")

    async def _process_task(self, task: Dict):
        """Process a queued task."""
        try:
            task_type = task['type']
            data = task['data']

            if task_type == 'generate_from_feed':
                has_new_articles = data.get('has_new_articles', True)
                self.logger.info(f"Processing generation task for {len(data['articles'])} articles (has_new_articles: {has_new_articles})")

                # Convert back to ArticleData objects
                articles = [ArticleData(**article) for article in data['articles']]

                result = await self.blog_service.generate_from_feed_items(articles, has_new_articles=has_new_articles)

                if result.get('success'):
                    self.logger.info(f"Blog generation successful: {result.get('file_path')}")
                    if result.get('published_path'):
                        self.logger.info(f"Blog post published to: {result.get('published_path')}")

                    # Enqueue another generation task to keep generating more blog posts
                    self.task_queue.enqueue('generate_from_feed', {
                        'articles': [],
                        'ingestion_result': None,
                        'has_new_articles': False
                    })
                else:
                    self.logger.error(f"Blog generation failed: {result.get('error')}")

            self.task_queue.complete_task(task['id'])

        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            # Mark as failed but don't complete
            task['status'] = 'failed'


class OneOffRunner:
    """Runner for single execution (manual mode)."""

    def __init__(self):
        self.feed_fetcher = FeedFetcher()
        self.rss_ingestor = RSSIngestor()
        self.blog_service = BlogGenerationService()
        self.logger = logger

    async def run(self):
        """Run one-off generation pipeline."""
        self.logger.info("Running one-off blog generation")

        try:
            # Fetch new articles
            new_articles = await self.feed_fetcher.fetch_new_feeds()

            if not new_articles:
                self.logger.info("No new articles to process")
                return

            # Ingest articles
            ingestion_result = await self.rss_ingestor.ingest_articles(new_articles)

            if 'error' in ingestion_result:
                self.logger.error(f"Ingestion failed: {ingestion_result['error']}")
                return

            # Generate blog
            result = await self.blog_service.generate_from_feed_items(new_articles)

            if result.get('success'):
                self.logger.info(f"One-off generation completed: {result.get('file_path')}")
            else:
                self.logger.error(f"One-off generation failed: {result.get('error')}")

        except Exception as e:
            self.logger.error(f"One-off run failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autoblog Continuous Generation Daemon")

    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['manual', 'daemon'],
        default=os.environ.get('AUTOBLOG_MODE', 'manual'),
        help='Mode to run in (default: manual, or set AUTOBLOG_MODE env var)'
    )

    # Daemon options
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Check interval in minutes for daemon mode (default: 30)'
    )

    args = parser.parse_args()

    if args.mode == 'manual':
        logger.info("Starting in MANUAL (one-off) mode")
        runner = OneOffRunner()
        asyncio.run(runner.run())
    else:
        logger.info("Starting in DAEMON (continuous) mode")
        daemon = ContinuousDaemon(check_interval_minutes=args.interval)
        daemon.start()


if __name__ == "__main__":
    main()
