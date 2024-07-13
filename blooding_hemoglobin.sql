--itemid:51222, 50811, g/dL
DROP TABLE IF EXISTS blooding_hemoglobin; CREATE TABLE blooding_hemoglobin AS 
SELECT
    MAX(subject_id) AS subject_id
    , MAX(hadm_id) AS hadm_id
    , MAX(charttime) AS charttime
    , le.specimen_id
    -- convert from itemid into a meaningful column
    , MAX(CASE WHEN itemid = 51222 or itemid = 50811 THEN valuenum ELSE NULL END) AS hemoglobin --g/dL

FROM mimiciv_hosp.labevents le
WHERE le.itemid IN
    (
        50811 -- hematocrit
        , 51222 -- hemoglobin
    )
    AND valuenum IS NOT NULL
    -- lab values cannot be 0 and cannot be negative
    AND valuenum > 0
GROUP BY le.specimen_id
;
