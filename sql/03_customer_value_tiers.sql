-- Customer value quartiles
WITH customer_spend AS (
    SELECT
        c.customer_unique_id,
        SUM(oi.price + oi.freight_value) AS total_spend,
        COUNT(DISTINCT o.order_id)       AS n_orders
    FROM orders o
    JOIN order_items oi ON o.order_id    = oi.order_id
    JOIN customers  c  ON o.customer_id  = c.customer_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY 1
)
SELECT *,
    NTILE(4) OVER (ORDER BY total_spend DESC) AS value_quartile
FROM customer_spend;
