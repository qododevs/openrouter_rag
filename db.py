from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, BigInteger, Text, select
from config import DATABASE_URL

Base = declarative_base()

class ChatContext(Base):
    __tablename__ = "chat_contexts"
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger, unique=True, nullable=False)
    context = Column(Text, default="")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_context(chat_id: int) -> str:
    async with async_session() as session:
        result = await session.execute(select(ChatContext).where(ChatContext.chat_id == chat_id))
        row = result.scalar_one_or_none()
        return row.context if row else ""

async def set_context(chat_id: int, context: str):
    async with async_session() as session:
        result = await session.execute(select(ChatContext).where(ChatContext.chat_id == chat_id))
        row = result.scalar_one_or_none()
        if row:
            row.context = context
        else:
            row = ChatContext(chat_id=chat_id, context=context)
            session.add(row)
        await session.commit()

async def clear_context(chat_id: int):
    await set_context(chat_id, "")