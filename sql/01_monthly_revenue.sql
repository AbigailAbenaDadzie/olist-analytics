-- Monthly revenue with running total (window function)
SELECT
    date_trunc('month', o.order_purchase_timestamp) AS month,
    SUM(oi.price + oi.freight_value)                AS revenue,
    SUM(SUM(oi.price + oi.freight_value)) OVER (
        ORDER BY date_trunc('month', o.order_purchase_timestamp)
    )                                                AS running_total
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY 1;
