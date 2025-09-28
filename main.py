import sys
import logging
import time
import subprocess
from datetime import datetime
import os
import argparse

#!/usr/bin/env python
# main.py - Business Analytics Pipeline Runner


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analytics_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BusinessAnalyticsPipeline")

# Define script paths - modify these to match your actual file locations
ETL_SCRIPT = "etl.py"
MBA_SCRIPT = "market_basket_analysis.py"
PED_SCRIPT = "price_elasticity.py"
DESC_SCRIPT = "descriptive_analytics.py"


def run_script(script_path, description):
    """Run a Python script and log its output."""
    logger.info(f"Starting {description}...")

    try:
        # Check if script exists
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            capture_output=True
        )

        # Log the output
        if result.stdout:
            logger.info(f"{description} output:\n{result.stdout}")

        logger.info(f"{description} completed successfully.")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}: {e}")
        if e.stdout:
            logger.info(f"{description} stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"{description} stderr:\n{e.stderr}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error running {description}: {str(e)}")
        return False


def main():
    """
    Main function to run the business analytics pipeline:
    ETL → Market Basket Analysis
    """
    parser = argparse.ArgumentParser(
        description="Business Analytics Pipeline Runner")
    parser.add_argument("--etl", action="store_true",
                        help="Run only the ETL process.")
    parser.add_argument("--mba", action="store_true",
                        help="Run only the Market Basket Analysis.")
    # parser.add_argument("--ped", action="store_true",
    #                     help="Run only the Price Elasticity analysis.")
    # parser.add_argument("--desc", action="store_true",
    #                     help="Run only the Descriptive Analytics.")
    args = parser.parse_args()

    start_time = time.time()
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Starting Business Analytics Pipeline - {run_date}")

    # Determine which parts of the pipeline to run
    run_all = not any([args.etl, args.mba])

    # Step 1: Extract, Transform, Load data
    if run_all or args.etl:
        if not run_script(ETL_SCRIPT, "ETL process"):
            logger.error("ETL process failed. Stopping pipeline.")
            return False

    # Step 2: Market Basket Analysis
    if run_all or args.mba:
        if not run_script(MBA_SCRIPT, "Market Basket Analysis"):
            logger.error("Market Basket Analysis failed. Stopping pipeline.")
            return False

    # Step 3: Price Elasticity of Demand (Temporarily disabled)
    # if run_all or args.ped:
    #     if not run_script(PED_SCRIPT, "Price Elasticity analysis"):
    #         logger.error(
    #             "Price Elasticity analysis failed. Stopping pipeline.")
    #         return False

    # Step 4: Descriptive Analytics (Temporarily disabled)
    # if run_all or args.desc:
    #     if not run_script(DESC_SCRIPT, "Descriptive Analytics"):
    #         logger.error("Descriptive Analytics failed. Stopping pipeline.")
    #         return False

    elapsed_time = time.time() - start_time
    logger.info(
        f"Business Analytics Pipeline completed successfully in {elapsed_time:.2f} seconds.")
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("Pipeline executed successfully!")
        sys.exit(0)
    else:
        print("Pipeline failed. Check logs for details.")
        sys.exit(1)
