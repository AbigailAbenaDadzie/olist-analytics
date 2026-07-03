-- Top sellers ranked by revenue within each product category
WITH seller_cat_revenue AS (
    SELECT
        p.product_category_name AS category,
        oi.seller_id,
        SUM(oi.price)           AS revenue,
        COUNT(*)                AS items_sold
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN sellers  s ON oi.seller_id  = s.seller_id
    WHERE p.product_category_name IS NOT NULL
    GROUP BY 1, 2
),
ranked AS (
    SELECT *,
        RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS rank_in_category
    FROM seller_cat_revenue
)
SELECT * FROM ranked WHERE rank_in_category <= 3
ORDER BY category, rank_in_category;
