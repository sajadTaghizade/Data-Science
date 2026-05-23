#!/usr/bin/env python3
"""
Ashpaz Project - Part 3
PySpark Batch + Real-Time Streaming Processing Layer
"""

import argparse
import os
from typing import Optional

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


ITEM_SCHEMA = T.StructType([
    T.StructField("item_id", T.StringType(), True),
    T.StructField("item_name", T.StringType(), True),
    T.StructField("name", T.StringType(), True),
    T.StructField("food_item", T.StringType(), True),
    T.StructField("unit_price", T.DoubleType(), True),
    T.StructField("quantity", T.IntegerType(), True),
])

ORDER_SCHEMA = T.StructType([
    T.StructField("order_id", T.StringType(), True),
    T.StructField("user_id", T.StringType(), True),
    T.StructField("restaurant_id", T.StringType(), True),
    T.StructField("restaurant_name", T.StringType(), True),
    T.StructField("restaurant_city", T.StringType(), True),
    T.StructField("cuisines", T.ArrayType(T.StringType()), True),
    T.StructField("phone_number", T.StringType(), True),
    T.StructField("request_online", T.BooleanType(), True),
    T.StructField("request_table", T.BooleanType(), True),
    T.StructField("order_time", T.StringType(), True),
    T.StructField("items", T.ArrayType(ITEM_SCHEMA), True),
    T.StructField("order_price", T.DoubleType(), True),
])


def create_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def parse_order_time(col):
    return F.coalesce(
        F.to_timestamp(col),
        F.to_timestamp(col, "yyyy-MM-dd'T'HH:mm:ss'Z'"),
        F.to_timestamp(col, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
        F.to_timestamp(col, "yyyy-MM-dd'T'HH:mm:ssX"),
        F.to_timestamp(col, "yyyy-MM-dd HH:mm:ss"),
    )


def item_name_expr():
    return F.coalesce(
        F.col("item.food_item"),
        F.col("item.item_name"),
        F.col("item.name"),
        F.col("item.item_id")
    )


def write_csv(df: DataFrame, path: str):
    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(path)
    )


def show_and_save(df: DataFrame, title: str, output_dir: str, rows: int = 20):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    df.show(rows, truncate=False)

    safe_title = (
        title.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

    write_csv(df, os.path.join(output_dir, safe_title))


def read_historical_data(spark: SparkSession, path: str) -> DataFrame:
    df = (
        spark.read
        .schema(ORDER_SCHEMA)
        .option("mode", "PERMISSIVE")
        .json(path)
    )

    if df.filter(F.col("order_id").isNotNull()).limit(1).count() == 0:
        df = (
            spark.read
            .schema(ORDER_SCHEMA)
            .option("multiLine", True)
            .option("mode", "PERMISSIVE")
            .json(path)
        )

    df = (
        df
        .filter(F.col("order_id").isNotNull())
        .withColumn("order_price", F.col("order_price").cast("double"))
        .withColumn("event_time", parse_order_time(F.col("order_time")))
    )

    return df


def top_ordered_items(orders: DataFrame) -> DataFrame:
    items = (
        orders
        .select(F.explode_outer("items").alias("item"))
        .filter(F.col("item").isNotNull())
        .withColumn("item_id", F.col("item.item_id"))
        .withColumn("item_name", item_name_expr())
        .withColumn("quantity", F.coalesce(F.col("item.quantity"), F.lit(1)))
    )

    return (
        items
        .groupBy("item_id", "item_name")
        .agg(
            F.sum("quantity").alias("total_quantity_ordered"),
            F.count("*").alias("order_lines_count")
        )
        .orderBy(
            F.desc("total_quantity_ordered"),
            F.desc("order_lines_count"),
            F.asc("item_name")
        )
    )


def highest_revenue_restaurants(orders: DataFrame) -> DataFrame:
    return (
        orders
        .groupBy("restaurant_id", "restaurant_name")
        .agg(
            F.round(F.sum("order_price"), 2).alias("total_revenue"),
            F.count("*").alias("total_orders"),
            F.round(F.avg("order_price"), 2).alias("avg_order_value")
        )
        .orderBy(F.desc("total_revenue"))
    )


def order_channel_leaders(orders: DataFrame) -> DataFrame:
    return (
        orders
        .groupBy("restaurant_id", "restaurant_name")
        .agg(
            F.sum(F.when(F.col("request_online") == True, 1).otherwise(0)).alias("online_orders"),
            F.sum(F.when(F.col("request_table") == True, 1).otherwise(0)).alias("dine_in_orders"),
            F.count("*").alias("total_orders")
        )
        .orderBy(
            F.desc("online_orders"),
            F.desc("dine_in_orders"),
            F.desc("total_orders")
        )
    )


def hourly_order_patterns(orders: DataFrame) -> DataFrame:
    return (
        orders
        .filter(F.col("event_time").isNotNull())
        .withColumn("hour", F.hour("event_time"))
        .groupBy("hour")
        .agg(F.count("*").alias("order_count"))
        .orderBy("hour")
    )


def peak_and_off_peak_hours(hourly: DataFrame) -> DataFrame:
    peak = (
        hourly
        .withColumn("type", F.lit("PEAK_HOUR"))
        .withColumn("rank", F.dense_rank().over(Window.orderBy(F.desc("order_count"))))
        .filter(F.col("rank") == 1)
        .drop("rank")
    )

    off_peak = (
        hourly
        .withColumn("type", F.lit("OFF_PEAK_HOUR"))
        .withColumn("rank", F.dense_rank().over(Window.orderBy(F.asc("order_count"))))
        .filter(F.col("rank") == 1)
        .drop("rank")
    )

    return peak.unionByName(off_peak).select("type", "hour", "order_count")


def revenue_distribution_by_city(orders: DataFrame) -> DataFrame:
    return (
        orders
        .groupBy("restaurant_city")
        .agg(
            F.round(F.sum("order_price"), 2).alias("total_revenue"),
            F.count("*").alias("total_orders"),
            F.round(F.avg("order_price"), 2).alias("avg_order_value")
        )
        .orderBy(F.desc("total_revenue"))
    )


def regional_favorite_foods(orders: DataFrame) -> DataFrame:
    city_items = (
        orders
        .select(
            "restaurant_city",
            F.explode_outer("items").alias("item")
        )
        .filter(F.col("restaurant_city").isNotNull())
        .filter(F.col("item").isNotNull())
        .withColumn("item_id", F.col("item.item_id"))
        .withColumn("item_name", item_name_expr())
        .withColumn("quantity", F.coalesce(F.col("item.quantity"), F.lit(1)))
        .groupBy("restaurant_city", "item_id", "item_name")
        .agg(F.sum("quantity").alias("total_quantity_ordered"))
    )

    ranked = (
        city_items
        .withColumn(
            "rank",
            F.row_number().over(
                Window.partitionBy("restaurant_city")
                .orderBy(F.desc("total_quantity_ordered"), F.asc("item_name"))
            )
        )
        .filter(F.col("rank") == 1)
        .drop("rank")
    )

    return ranked.orderBy("restaurant_city")


def run_batch(historical_path: str, output_dir: str):
    spark = create_spark("Ashpaz-Part3-Batch")
    os.makedirs(output_dir, exist_ok=True)

    try:
        orders = read_historical_data(spark, historical_path)

        print("\nLoaded historical data:")
        orders.printSchema()
        orders.show(5, truncate=False)

        result1 = top_ordered_items(orders)
        show_and_save(result1, "Top Ordered Items", output_dir)

        result2 = highest_revenue_restaurants(orders)
        show_and_save(result2, "Highest Revenue Restaurants", output_dir)

        result3 = order_channel_leaders(orders)
        show_and_save(result3, "Order Channel Leaders Online vs Dine In", output_dir)

        result4 = hourly_order_patterns(orders)
        show_and_save(result4, "Hourly Order Patterns", output_dir, rows=24)

        result5 = peak_and_off_peak_hours(result4)
        show_and_save(result5, "Peak and Off Peak Hours", output_dir)

        result6 = revenue_distribution_by_city(orders)
        show_and_save(result6, "Revenue Distribution By City", output_dir)

        result7 = regional_favorite_foods(orders)
        show_and_save(result7, "Regional Favorite Foods", output_dir, rows=100)

    finally:
        spark.stop()



def read_stream_from_kafka(
    spark: SparkSession,
    bootstrap_servers: str,
    input_topic: str
) -> DataFrame:

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", input_topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw
        .select(
            F.col("timestamp").alias("kafka_timestamp"),
            F.col("key").cast("string").alias("kafka_key"),
            F.col("value").cast("string").alias("json_value")
        )
        .select(
            "kafka_timestamp",
            "kafka_key",
            "json_value",
            F.from_json(F.col("json_value"), ORDER_SCHEMA).alias("data")
        )
        .select(
            "kafka_timestamp",
            "kafka_key",
            "json_value",
            "data.*"
        )
        .filter(F.col("order_id").isNotNull())
        .withColumn("order_price", F.col("order_price").cast("double"))
        .withColumn(
            "event_time",
            F.coalesce(
                parse_order_time(F.col("order_time")),
                F.col("kafka_timestamp")
            )
        )
    )

    return parsed


def write_batch_to_kafka(
    batch_df: DataFrame,
    batch_id: int,
    bootstrap_servers: str,
    topic: str,
    title: str
):
    if batch_df.isEmpty(): 
        return

    print("\n" + "=" * 90)
    print(f"{title} | Batch ID: {batch_id}")
    print("=" * 90)
    batch_df.show(20, truncate=False)

    kafka_df = batch_df.select(
        F.to_json(F.struct(*[F.col(c) for c in batch_df.columns])).alias("value")
    )

    (
        kafka_df.write
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", topic)
        .save()
    )


def start_cuisine_analytics(
    orders: DataFrame,
    bootstrap_servers: str,
    output_topic: str,
    checkpoint_dir: str,
    trigger_time: str
):
    cuisine_orders = (
        orders
        .withWatermark("event_time", "2 hours")
        .select(
            F.window("event_time", "1 minute", "20 seconds").alias("window"),
            F.explode_outer("cuisines").alias("cuisine"),
            F.col("order_price")
        )
        .filter(F.col("cuisine").isNotNull())
    )

    cuisine_analytics = (
        cuisine_orders
        .groupBy("window", "cuisine")
        .agg(
            F.round(F.sum("order_price"), 2).alias("total_revenue"),
            F.round(F.avg("order_price"), 2).alias("average_order_value"),
            F.count("*").alias("total_orders")
        )
        .withColumn("insight_type", F.lit("CUISINE_REVENUE_AND_AOV"))
    )

    return (
        cuisine_analytics.writeStream
        .outputMode("update")
        .foreachBatch(
            lambda df, batch_id: write_batch_to_kafka(
                df,
                batch_id,
                bootstrap_servers,
                output_topic,
                "Real-Time Cuisine Revenue and AOV"
            )
        )
        .option("checkpointLocation", os.path.join(checkpoint_dir, "cuisine_analytics"))
        .trigger(processingTime=trigger_time)
        .start()
    )


def start_geo_fraud_detection(
    orders: DataFrame,
    bootstrap_servers: str,
    fraud_topic: str,
    checkpoint_dir: str,
    trigger_time: str
):
    geo_alerts = (
        orders
        .filter(F.col("user_id").isNotNull())
        .filter(F.col("restaurant_city").isNotNull())
        .withWatermark("event_time", "2 hours")
        .groupBy(
            F.window("event_time", "30 minutes", "5 minutes").alias("window"),
            F.col("user_id")
        )
        .agg(
            F.count("*").alias("order_count"),
            F.min("restaurant_city").alias("city_a"),
            F.max("restaurant_city").alias("city_b"),
            F.min("event_time").alias("first_order_time"),
            F.max("event_time").alias("last_order_time"),
            F.min("order_id").alias("sample_order_id_1"),
            F.max("order_id").alias("sample_order_id_2")
        )
        .filter(F.col("city_a") != F.col("city_b"))
        .withColumn("alert_type", F.lit("GEOGRAPHICAL_IMPOSSIBILITY"))
        .withColumn(
            "reason",
            F.concat(
                F.lit("Same user ordered from different cities within 30 simulated minutes: "),
                F.col("city_a"),
                F.lit(" and "),
                F.col("city_b")
            )
        )
        .select(
            "alert_type",
            "reason",
            "user_id",
            "window",
            "order_count",
            "city_a",
            "city_b",
            "first_order_time",
            "last_order_time",
            "sample_order_id_1",
            "sample_order_id_2"
        )
    )

    return (
        geo_alerts.writeStream
        .outputMode("update")
        .foreachBatch(
            lambda df, batch_id: write_batch_to_kafka(
                df,
                batch_id,
                bootstrap_servers,
                fraud_topic,
                "Fraud Alert - Geographical Impossibility"
            )
        )
        .option("checkpointLocation", os.path.join(checkpoint_dir, "geo_fraud"))
        .trigger(processingTime=trigger_time)
        .start()
    )


def start_velocity_fraud_detection(
    orders: DataFrame,
    bootstrap_servers: str,
    fraud_topic: str,
    checkpoint_dir: str,
    trigger_time: str
):
    velocity_alerts = (
        orders
        .filter(F.col("user_id").isNotNull())
        .withWatermark("event_time", "2 hours")
        .groupBy(
            F.window("event_time", "60 minutes", "5 minutes").alias("window"),
            F.col("user_id")
        )
        .agg(
            F.count("*").alias("order_count"),
            F.min("event_time").alias("first_order_time"),
            F.max("event_time").alias("last_order_time"),
            F.min("order_id").alias("sample_first_order_id"),
            F.max("order_id").alias("sample_last_order_id")
        )
        .filter(F.col("order_count") > 5)
        .withColumn("alert_type", F.lit("VELOCITY_ORDER_SPAM"))
        .withColumn(
            "reason",
            F.concat(
                F.lit("User placed "),
                F.col("order_count").cast("string"),
                F.lit(" orders within 60 simulated minutes.")
            )
        )
        .select(
            "alert_type",
            "reason",
            "user_id",
            "window",
            "order_count",
            "first_order_time",
            "last_order_time",
            "sample_first_order_id",
            "sample_last_order_id"
        )
    )

    return (
        velocity_alerts.writeStream
        .outputMode("update")
        .foreachBatch(
            lambda df, batch_id: write_batch_to_kafka(
                df,
                batch_id,
                bootstrap_servers,
                fraud_topic,
                "Fraud Alert - Velocity Check"
            )
        )
        .option("checkpointLocation", os.path.join(checkpoint_dir, "velocity_fraud"))
        .trigger(processingTime=trigger_time)
        .start()
    )


def run_streaming(
    bootstrap_servers: str,
    input_topic: str,
    fraud_topic: str,
    insights_topic: str,
    checkpoint_dir: str,
    trigger_time: str
):
    spark = create_spark("Ashpaz-Part3-Streaming")
    os.makedirs(checkpoint_dir, exist_ok=True)

    orders = read_stream_from_kafka(
        spark=spark,
        bootstrap_servers=bootstrap_servers,
        input_topic=input_topic
    )

    query1 = start_cuisine_analytics(
        orders=orders,
        bootstrap_servers=bootstrap_servers,
        output_topic=insights_topic,
        checkpoint_dir=checkpoint_dir,
        trigger_time=trigger_time
    )

    query2 = start_geo_fraud_detection(
        orders=orders,
        bootstrap_servers=bootstrap_servers,
        fraud_topic=fraud_topic,
        checkpoint_dir=checkpoint_dir,
        trigger_time=trigger_time
    )

    query3 = start_velocity_fraud_detection(
        orders=orders,
        bootstrap_servers=bootstrap_servers,
        fraud_topic=fraud_topic,
        checkpoint_dir=checkpoint_dir,
        trigger_time=trigger_time
    )

    print("\nStreaming started successfully.")
    print(f"Input topic: {input_topic}")
    print(f"Fraud topic: {fraud_topic}")
    print(f"Insights topic: {insights_topic}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("Press Ctrl+C to stop.\n")

    try:
        spark.streams.awaitAnyTermination()
    finally:
        query1.stop()
        query2.stop()
        query3.stop()
        spark.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Ashpaz Part 3 PySpark Code")

    parser.add_argument(
        "--mode",
        choices=["batch", "streaming"],
        required=True,
        help="Run batch or streaming part"
    )

    parser.add_argument(
        "--historical",
        default="historical_data.json",
        help="Path to historical_data.json"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/part3_batch",
        help="Output folder for batch CSV files"
    )

    parser.add_argument(
        "--bootstrap",
        default="localhost:9092",
        help="Kafka bootstrap server"
    )

    parser.add_argument(
        "--input-topic",
        default="ashpaz.valid",
        help="Kafka topic for valid orders"
    )

    parser.add_argument(
        "--fraud-topic",
        default="orders.fraud_alerts",
        help="Kafka topic for fraud alerts"
    )

    parser.add_argument(
        "--insights-topic",
        default="orders.cuisine_insights",
        help="Kafka topic for cuisine insights"
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/part3_streaming",
        help="Checkpoint directory for Spark Streaming"
    )

    parser.add_argument(
        "--trigger",
        default="20 seconds",
        help="Micro-batch trigger interval"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "batch":
        run_batch(
            historical_path=args.historical,
            output_dir=args.output_dir
        )

    elif args.mode == "streaming":
        run_streaming(
            bootstrap_servers=args.bootstrap,
            input_topic=args.input_topic,
            fraud_topic=args.fraud_topic,
            insights_topic=args.insights_topic,
            checkpoint_dir=args.checkpoint_dir,
            trigger_time=args.trigger
        )


if __name__ == "__main__":
    main()