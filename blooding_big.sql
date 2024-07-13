--itemid:51222, 50811, g/dL
-- with diffs as(
-- SELECT subject_id, hadm_id,
-- charttime,
-- hemoglobin,
-- hemoglobin - LAG(hemoglobin) OVER (PARTITION BY hadm_id ORDER BY charttime) as diff
-- FROM mimiciv_derived.complete_blood_count le
-- WHERE charttime >= DATETIME_SUB(charttime, INTERVAL '6' HOUR) 
-- and  charttime <= DATETIME_ADD(charttime, INTERVAL '1' DAY)
-- 
-- --GROUP BY subject_id,hadm_id,charttime,hemoglobin
-- )
-- SELECT
-- 		ie.subject_id, ie.stay_id,charttime, hemoglobin,
--     MAX(diff) OVER (PARTITION BY ie.stay_id ORDER BY charttime RANGE BETWEEN 24 PRECEDING AND CURRENT ROW) AS max_diff,
--     MIN(diff) OVER (PARTITION BY ie.stay_id ORDER BY charttime RANGE BETWEEN 24 PRECEDING AND CURRENT ROW) AS min_diff
-- FROM mimiciv_icu.icustays ie
--   LEFT JOIN diffs le
--       ON le.subject_id = ie.subject_id
-- GROUP BY ie.stay_id, diff,charttime,hemoglobin
-- ORDER BY ie.subject_id, ie.stay_id
-- 
--DROP TABLE IF EXISTS blooding_delta; CREATE TABLE blooding_delta AS 
SELECT
    subject_id,hadm_id,charttime,hemoglobin,
    round(cast (MAX(diff) as numeric),2) AS max_diff,
    round(cast (MAX(diff) as numeric),2) AS min_diff
FROM (
    SELECT
		    mimiciv_derived.blooding_hemoglobin.subject_id,mimiciv_derived.blooding_hemoglobin.hadm_id,
        mimiciv_derived.blooding_hemoglobin.charttime,hemoglobin,
        hemoglobin - LAG(hemoglobin) OVER w AS diff,
        charttime - MOD(EXTRACT(EPOCH FROM charttime)::integer, 86400) * INTERVAL '1 second' AS period_start
    FROM
        mimiciv_derived.blooding_hemoglobin
		LEFT JOIN mimiciv_hosp.admissions on mimiciv_hosp.admissions.hadm_id = mimiciv_derived.blooding_hemoglobin.hadm_id
    WHERE
        charttime BETWEEN admittime AND dischtime
    WINDOW
        w AS (PARTITION BY blooding_hemoglobin.subject_id ORDER BY charttime ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)
) AS subquery
GROUP BY
    subject_id,hadm_id,charttime,hemoglobin
ORDER BY subject_id,hadm_id








