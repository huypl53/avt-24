import logging
from app.db.setting import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mode():
    assert settings.mode == "test"


def test_connection():
    logging.info(settings.sql_url.unicode_string())
