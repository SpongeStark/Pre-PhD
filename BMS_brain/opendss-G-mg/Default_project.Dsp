V2.0;;;
1	ELEMENT	 	NODE	0	233	0	sbus	14	233	14	233	 	 	 	 
2	ELEMENT	 	NODE	413	233	0	slv	427	233	427	233	 	 	 	 
3	ELEMENT	 	NODE	786	233	0	pcc	800	233	800	233	 	 	 	 
4	ELEMENT	 	NODE	1159	233	0	n_dc_bus	1173	233	1173	233	 	 	 	 
5	ELEMENT	 	NODE	1159	0	0	n_ev_dc	1173	0	1173	0	 	 	 	 
6	ELEMENT	 	NODE	1532	233	0	n_bat_dc	1546	233	1546	233	 	 	 	 
7	ELEMENT	 	NODE	1159	466	0	n_pv_dc	1173	466	1173	466	 	 	 	 
8	ELEMENT	 	NODE	786	0	0	n_load	800	0	800	0	 	 	 	 
9	LINE	Phases=3	 	0	0	0	line_0	427	233	800	233	slv.1.2.3	pcc.1.2.3	427,233;800,233;800,233--2	 
10	LINE	Phases=3	 	0	0	0	line_load	800	233	800	0	pcc.1.2.3	n_load.1.2.3	800,233;800,0;800,0--2	 
11	LINE	Phases=3	 	0	0	0	line_reactor	800	233	1173	233	pcc.1.2.3	n_dc_bus.1.2.3	800,233;1173,233;1173,233--2	 
12	LINE	Phases=3	 	0	0	0	line_pv_dc	1173	233	1173	466	n_dc_bus.1.2.3	n_pv_dc.1.2.3	1173,233;1173,466;1173,466--2	 
13	LINE	Phases=3	 	0	0	0	line_bat_dc	1173	233	1546	233	n_dc_bus.1.2.3	n_bat_dc.1.2.3	1173,233;1546,233;1546,233--2	 
14	LINE	Phases=3	 	0	0	0	line_ev_dc	1173	233	1173	0	n_dc_bus.1.2.3	n_ev_dc.1.2.3	1173,233;1173,0;1173,0--2	 
15	SYSTEM	3	MAIN_NODE	14	233	0	 	14	233	20000	20000	sbus.1.2.3	 	 	 
16	ELEMENT	Phases=3--	TRANSFORMER_YY	14	233	0	xfmr_0	14	233	427	233	sbus.1.2.3	slv.1.2.3	 	 
