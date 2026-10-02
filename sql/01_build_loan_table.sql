CREATE OR REPLACE TABLE loans AS
SELECT
    id,
    strptime(issue_d, '%b-%Y')                       AS issue_date,
    CASE WHEN loan_status = 'Charged Off' THEN 1
         ELSE 0 END                                  AS default_flag,
    loan_amnt,
    int_rate,
    grade,
    emp_length,
    home_ownership,
    annual_inc,
    dti,
    fico_range_low,
    revol_util,
    delinq_2yrs,
    inq_last_6mths,
    open_acc,
    pub_rec,
    purpose,
    addr_state
FROM read_csv_auto('{RAW_FILE}', ignore_errors = true)
WHERE loan_status IN ('Fully Paid', 'Charged Off')
  AND term LIKE '%36%';
