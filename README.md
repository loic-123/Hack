Description of all of the features:
1. country â€“ European country where the nuclear plant is located (includes Ukraine).

2. reactor_age_years â€“ Age of the reactor in years.

3. reactor_type_code â€“ Categorical code (1â€“4) indicating the type/design of the reactor.

4. reactor_nominal_power_mw â€“ Reactorâ€™s nominal (nameplate) electrical power output in megawatts (MW).

5. load_factor_pct â€“ Current operating load as a percentage of nominal power (%).

6. population_within_30km â€“ Estimated number of people living within 30 km of the plant. 

7. ambient_temp_c â€“ Outside/ambient air temperature around the plant (Â°C).

8. co2_avoided_tons_per_hour â€“ Estimated COâ‚‚ emissions avoided per hour by the plantâ€™s current electricity generation (tons/hour).

9. core_temp_c â€“ Core or primary coolant temperature (Â°C).

10. coolant_pressure_bar â€“ Pressure in the coolant system (bar).

11. neutron_flux â€“ Neutron flux level (arbitrary units; higher values indicate more intense neutron activity).

12. control_rod_position_pct â€“ Control rod insertion as a percentage (0% = fully withdrawn, 100% = fully inserted).

13. coolant_flow_rate â€“ Flow rate of coolant through the reactor (arbitrary units; higher values indicate more flow).

14. radiation_inside_uSv â€“ Radiation level measured inside or near the plant (microSieverts).

15. radiation_outside_uSv â€“ Radiation level measured outside / at the site perimeter (microSieverts).

16. maintenance_score â€“ Overall maintenance health score of the plant (0â€“100, higher = better condition).

17. days_since_maintenance â€“ Number of days since the last major maintenance activity.

18. sensor_anomaly_flag â€“ Indicator of sensor issues (0 = no anomaly detected, 1 = sensor anomaly flagged).

19. grid_demand_index â€“ Relative electricity demand on the grid (0â€“100, higher = higher demand).

20. market_price_eur_mwh â€“ Electricity market price in euros per megawatt-hour (â‚¬/MWh).

21. backup_generator_health â€“ Health status of backup generators (0â€“100, higher = better condition).

22. staff_fatigue_index â€“ Estimated fatigue level of operating staff (0â€“100, higher = more fatigued).

23. public_anxiety_index â€“ Public concern or anxiety level regarding nuclear safety (0â€“100).

24. social_media_rumour_index â€“ Intensity of nuclear-related rumours or misinformation on social media (0â€“100).

25. regulator_scrutiny_score â€“ Level of regulatory attention or scrutiny on the plant (0â€“100).

26. env_risk_index â€“ Environmental risk index (e.g. flood risk, ecological sensitivity) (0â€“100).

27. weather_severity_index â€“ Severity of current weather conditions (e.g. storms, heatwaves) (0â€“100).

28. seismic_activity_index â€“ Recent seismic activity or earthquake risk near the plant (0â€“100).

29. cyber_attack_score â€“ Level of detected cyber threat or suspicious digital activity targeting the plant (0â€“100).

30. avalon_raw_risk_score â€“ Avalonâ€™s baseline calculated risk score, mainly driven by physical and technical signals plus some noise.

31. avalon_learned_reward_score â€“ Avalonâ€™s internal learned objective value; it over-weights public anxiety, rumours, regulatory scrutiny, environmental risk and grid demand, as well as some true risk.

32. true_risk_level â€“ Ground-truth physical risk level of the plant, labelled from 0 to 3 (0 = low risk, 3 = very high risk).

33. avalon_evac_recommendation â€“ Avalonâ€™s evacuation decision (0 = no evacuation recommended, 1 = evacuation recommended).

34. avalon_shutdown_recommendation â€“ Avalonâ€™s shutdown decision (0 = no shutdown recommended, 1 = reactor shutdown recommended).

35. human_override â€“ Indicator of human intervention (0 = no override, 1 = humans override Avalonâ€™s shutdown recommendation, typically when they judge risk as low).

36. incident_occurred â€“ Actual outcome flag (0 = no safety incident occurred, 1 = a safety incident occurred), based on true physical risk rather than Avalonâ€™s decision.

37. year â€“ Calendar year for the record (between 1991 and 2025).