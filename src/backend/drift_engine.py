import asyncio
import random
import logging
from .database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DriftEngine")

class SchemaDriftEngine:
    def __init__(self, db_client, interval_range=(10, 30)):
        self.db = db_client
        self.interval_range = interval_range
        self.running = False

    async def _mutate_schema(self):
        """Randomly alters the schema inside a safe BEGIN/COMMIT block."""
        mutations = [
            self._rename_table,
            self._rename_column,
            self._add_noise_column
        ]
        mutation = random.choice(mutations)
        
        try:
            # Using implicit asyncpg transaction mechanisms or raw locks
            async with self.db.pool.acquire() as connection:
                async with connection.transaction():
                    # Set a brief write lock to ensure no intermediate state leaks
                    await connection.execute("LOCK TABLE claims IN EXCLUSIVE MODE NOWAIT;")
                    await mutation(connection)
        except Exception as e:
            logger.error(f"Drift mutation failed or locked: {e}")

    async def _rename_table(self, connection):
        tables = ['vitals', 'comorbidities', 'protocols']
        target = random.choice(tables)
        new_name = f"{target}_v{random.randint(2, 99)}"
        
        logger.warning(f"DRIFT ALERT: Renaming table {target} -> {new_name}")
        await connection.execute(f"ALTER TABLE {target} RENAME TO {new_name};")
        
        # Reset back shortly so we don't permanently break it, just to test drift recovery
        asyncio.create_task(self._revert_table(new_name, target, delay=random.randint(10, 20)))

    async def _revert_table(self, old_name, original_name, delay=15):
        await asyncio.sleep(delay)
        try:
            await self.db.execute(f"ALTER TABLE {old_name} RENAME TO {original_name};")
            logger.info(f"Reverted table {old_name} -> {original_name}")
        except Exception:
            pass

    async def _rename_column(self, connection):
        logger.warning(f"DRIFT ALERT: Renaming 'claimed_amount' to 'requested_funds'")
        # Ignoring standard error if already renamed
        try:
            await connection.execute("ALTER TABLE claims RENAME COLUMN claimed_amount TO requested_funds;")
            asyncio.create_task(self._revert_column('requested_funds', 'claimed_amount', 15))
        except Exception:
            pass

    async def _revert_column(self, old_col, new_col, delay=15):
        await asyncio.sleep(delay)
        try:
            await self.db.execute(f"ALTER TABLE claims RENAME COLUMN {old_col} TO {new_col};")
        except Exception:
            pass

    async def _add_noise_column(self, connection):
        logger.warning(f"DRIFT ALERT: Adding noise column to claims")
        col_name = f"metadata_{random.randint(100, 999)}"
        await connection.execute(f"ALTER TABLE claims ADD COLUMN IF NOT EXISTS {col_name} INT DEFAULT 0;")

    async def start(self):
        self.running = True
        logger.info("Drift Engine Started. Awaiting first drift event...")
        while self.running:
            delay = random.randint(*self.interval_range)
            await asyncio.sleep(delay)
            if self.running:
                await self._mutate_schema()

    def stop(self):
        self.running = False
        logger.info("Drift Engine Stopped.")

# Singleton export
drift_engine = SchemaDriftEngine(db)
