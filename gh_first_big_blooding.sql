
with tp1 as (
SELECT DISTINCT
	mimiciv_derived.gh.stay_id,
	mimiciv_derived.gh.subject_id,
	mimiciv_derived.gh.hadm_id,
	blooding_delta.charttime as bd_time,
	rbc_transfusion.charttime as rt_time,
	mimiciv_derived.gh.icu_intime as icu_intime
FROM
	mimiciv_derived.gh
	LEFT JOIN mimiciv_derived.blooding_delta ON gh.hadm_id = blooding_delta.hadm_id 
	    AND blooding_delta.charttime <= DATETIME_ADD ( mimiciv_derived.gh.icu_intime, INTERVAL '1' DAY )
	  --AND blooding_delta.charttime >= mimiciv_derived.gh.icu_intime 
	LEFT JOIN rbc_transfusion ON gh.stay_id = rbc_transfusion.stay_id 
	  AND rbc_transfusion.charttime <= DATETIME_ADD ( mimiciv_derived.gh.icu_intime, INTERVAL '1' DAY )
		-- AND rbc_transfusion.charttime >= mimiciv_derived.gh.icu_intime	 
WHERE
	max_diff >= 2 
	OR rbc_transfusion.amount >= 600 --GROUP BY mimiciv_derived.gh.stay_id,mimiciv_derived.gh.subject_id,mimiciv_derived.gh.hadm_id
	)
	,tp2 as (
	SELECT DISTINCT stay_id, subject_id,hadm_id,
	1 as excluded_label 
	FROM tp1 
 	WHERE 
	-- bd_time >= icu_intime
	--and rt_time >= icu_intime
  bd_time <= DATETIME_ADD (icu_intime, INTERVAL '1' DAY ) 
  or rt_time <= DATETIME_ADD (icu_intime, INTERVAL '1' DAY ) 
)
SELECT DISTINCT mimiciv_derived.gh.*
 FROM mimiciv_derived.gh
 left JOIN tp2 ON gh.hadm_id = tp2.hadm_id 
 and gh.subject_id = tp2.subject_id 
 and gh.stay_id = tp2.stay_id
 WHERE excluded_label is null 