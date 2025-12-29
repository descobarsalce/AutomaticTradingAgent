# Local Database Setup (macOS)

Use this guide to run the app locally on macOS when your previous hosted SQL instance is unavailable.

## Default: SQLite (no extra install required)
- The project falls back to a local **SQLite** file when `DATABASE_URL` is not set.
- Nothing to install with Homebrew. Just run the app and it will create `trading_data.db` in the project root.
- Good for demos and single-user testing. No network setup needed.

## Optional: PostgreSQL via Homebrew
If you want a heavier-duty database (shared usage, better concurrency), install PostgreSQL:

```bash
brew install postgresql@16
brew services start postgresql@16

# Initialize a database and create a user/db
initdb ~/Library/Application\ Support/Postgres
createuser trading_user --pwprompt
createdb trading_db -O trading_user
```

Then export `DATABASE_URL` so the app uses PostgreSQL instead of SQLite:

```bash
export DATABASE_URL="postgresql+psycopg2://trading_user:<your-password>@localhost:5432/trading_db"
```

## Verifying connectivity
- From the project root, run a quick check that the database is reachable:

```bash
python - <<'PY'
from src.utils.db_config import db_config
engine = db_config.engine
with engine.connect() as conn:
    conn.execute("SELECT 1")
print("Database connection OK")
PY
```

## Switching back to SQLite
- Unset `DATABASE_URL` or leave it empty and the app will automatically use the local `trading_data.db` file.
- You do **not** need Homebrew or PostgreSQL for SQLite mode.

## Notes
- Homebrew commands require the Xcode command-line tools (`xcode-select --install`) the first time you use brew.
- If port 5432 is busy, change the PostgreSQL port in `postgresql.conf` and update `DATABASE_URL` accordingly.
