#!/usr/bin/env python3
"""
Export training dataset to CSV for offline evaluation.

Usage:
    python scripts/export_training_data.py

Exports to /tmp/training_data.csv with all features.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_async_session
from app.features.engineering import FeatureEngineer


async def main():
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        print("Building training dataset...")
        df = await fe.build_training_dataset()

        output_path = "/tmp/training_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} samples to {output_path}")
        print(f"Columns: {list(df.columns)}")
        break


if __name__ == "__main__":
    asyncio.run(main())
