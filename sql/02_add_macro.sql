CREATE OR REPLACE TABLE loans_macro AS
WITH state_macro AS (
    SELECT
        state,
        date,
        state_unemp,
        state_unemp - LAG(state_unemp, 12) OVER (
            PARTITION BY state ORDER BY date
        ) AS state_unemp_chg_12m
    FROM macro_state
)
SELECT
    l.*,
    n.unemployment_rate,
    n.fed_funds_rate,
    s.state_unemp,
    s.state_unemp_chg_12m
FROM loans l
LEFT JOIN macro_national n
    ON date_trunc('month', l.issue_date) - INTERVAL 1 MONTH = n.date
LEFT JOIN state_macro s
    ON l.addr_state = s.state
   AND date_trunc('month', l.issue_date) - INTERVAL 1 MONTH = s.date;
