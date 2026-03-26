import schedule
import time
import os

def job():
    print("Running batch prediction...")
    os.system("python src/batch_predict.py")

# Run for every 10 seconds for demo purpose
schedule.every(10).seconds.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)