-- Loss outcomes for charged-off 36-month loans.
-- These are realized outcomes, used only to measure EAD and LGD,
-- never as model features.
SELECT
    CAST(id AS VARCHAR)                     AS id,
    funded_amnt,
    funded_amnt - total_rec_prncp           AS ead,
    recoveries - collection_recovery_fee    AS net_recovery
FROM read_csv_auto('{RAW_FILE}', ignore_errors = true)
WHERE loan_status = 'Charged Off'
  AND term LIKE '%36%';
