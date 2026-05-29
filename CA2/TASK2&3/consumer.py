import json
from confluent_kafka import Consumer, Producer

KAFKA_BROKER = 'localhost:9092'
INPUT_TOPIC = 'ashpaz.order'
VALID_TOPIC = 'ashpaz.valid'
ERROR_TOPIC = 'ashpaz.error_log'

def delivery_report(err, msg):
    if err is not None:
        print(f'Delivery failed: {err}')

def validate_order(order):
    errors = []
    
    phone = order.get('phone_number', '')
    if not (phone.startswith('+91') or phone.startswith('080')):
        errors.append('INVALID_PHONE')
        
    request_online = order.get('request_online', False)
    request_table = order.get('request_table', False)
    if request_online and request_table:
        errors.append('ORDER_MODE_CONFLICT')
        
    order_price = order.get('order_price', 0.0)
    items = order.get('items', [])
    calculated_price = sum(item.get('unit_price', 0) * item.get('quantity', 0) for item in items)
    
    if abs(order_price - calculated_price) > 0.01: 
        errors.append('PRICE_MISMATCH')
        
    return errors

def main():
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'ashpaz_validator_group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe([INPUT_TOPIC])

    producer = Producer({'bootstrap.servers': KAFKA_BROKER})

    print(f"Listening to '{INPUT_TOPIC}'...")

    try:
        while True:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
            if msg.error():
                print(f"Error: {msg.error()}")
                continue

            try:
                order_data = json.loads(msg.value().decode('utf-8'))
            except json.JSONDecodeError:
                continue
            
            error_list = validate_order(order_data)

            if not error_list:
                producer.produce(
                    VALID_TOPIC, 
                    value=json.dumps(order_data).encode('utf-8'),
                    callback=delivery_report
                )
                print(f"[VALID] Order {order_data.get('order_id')}")
            else:
                error_type = "MULTI" if len(error_list) > 1 else error_list[0]
                error_payload = {
                    "order_id": order_data.get("order_id"),
                    "error_type": error_type,
                    "error_reason": error_list,
                    "original_order": order_data
                }
                producer.produce(
                    ERROR_TOPIC, 
                    value=json.dumps(error_payload).encode('utf-8'),
                    callback=delivery_report
                )
                print(f"[INVALID] Order {order_data.get('order_id')} - Reason: {error_type}")
            
            producer.poll(0)

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        producer.flush()

if __name__ == '__main__':
    main()