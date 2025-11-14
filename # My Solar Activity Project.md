#  Solar Activity Project

I’ve used NOAA solar data to visualize sunspot activity over time.  
Data was collected from [NOAA’s observed solar cycle indices](https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json), and plotted to reveal the Sun’s behavior.  
A predictive dataset from the same source was merged with the observed data to create a unified graph.  


---

##  ETL Breakdown

### Extract
- Pulled data from two NOAA JSON APIs  
- Handled empty responses and API errors 

### Transform
- Standardized column names across datasets  
- Converted date strings to datetime objects  
- Added source labels for tracking  
- Merged datasets via concatenation  

### Load
- Stored data in pandas DataFrames  
- Prepared for visualization and further analysis  

---

##  Insights from the Graph

- Sunspot activity was lowest between 1792 and 1833  
- The peak occurred in 1958, marking the most sunspots on record  
- We are currently in a solar minimum  

---

##  Why This Matters

Accurate solar behavior prediction could enable protective measures during solar maximums—shielding satellites and Earth’s infrastructure from geomagnetic storms.  

This project is aimed at astronomers and space weather analysts seeking deeper insight into solar cycles and their implications.

