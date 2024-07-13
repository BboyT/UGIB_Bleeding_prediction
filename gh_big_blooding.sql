SELECT DISTINCT
   mimiciv_derived.gh.stay_id
	,mimiciv_derived.gh.subject_id,
	mimiciv_derived.gh.hadm_id,
	1 AS hospital_big_bleeding 
FROM
	mimiciv_derived.gh
	LEFT JOIN mimiciv_derived.blooding_delta ON gh.hadm_id = blooding_delta.hadm_id 
	AND blooding_delta.charttime < mimiciv_derived.gh.icu_intime --DATETIME_ADD ( mimiciv_derived.gh.icu_intime, INTERVAL '1' DAY )  
	LEFT JOIN rbc_transfusion ON gh.stay_id = rbc_transfusion.stay_id 
	AND rbc_transfusion.charttime < mimiciv_derived.gh.icu_intime  --DATETIME_ADD ( mimiciv_derived.gh.icu_intime, INTERVAL '1' DAY ) 
		
WHERE
	max_diff >= 2 
	OR rbc_transfusion.amount >= 600 --GROUP BY mimiciv_derived.gh.stay_id,mimiciv_derived.gh.subject_id,mimiciv_derived.gh.hadm_id