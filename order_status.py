import random, time

STATUSES = ["processing","packed","shipped","out for delivery","delivered","returned"]

def get_status(order_id: str):
    random.seed(hash(order_id) % 10_000)
    return {
        "orderId": order_id,
        "status": random.choice(STATUSES),
        "etaDays": random.randint(0, 5),
        "lastUpdated": int(time.time())
    }
