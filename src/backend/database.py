import os
import asyncpg

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgres://postgres:postgres@localhost:5432/panacea"
)

class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(DATABASE_URL)
        print("Database connected.")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            print("Database disconnected.")

    async def fetch(self, query: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
            
    async def fetchrow(self, query: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
            
    async def execute(self, query: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)

db = Database()
