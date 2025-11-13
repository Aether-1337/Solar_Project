 # My Solar Activity Project

 * I have used the NOAA solor data to plot sun spot activity over time 
 * I collected the data from https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json then plotted it onto a grpah to help visulise the suns behaviour 
 * There was also a predictive data set included on the website, this is represented on the graph - I mergred the observed data set with the predicted data set to create a single graph. 
 * Using AI I have created a code and made a brach within the repository that uses the historical data and machine learning to make a new model.  
 * The results that were fed back are significantly differnt than the current models prediction. 

 ### Extract
- Data extracted from two NOAA JSON APIs
- Handles API response errors and empty responses

### Transform
- Standardized column names between datasets
- Converted date strings to datetime objects
- Added source labels for data tracking
- Combined datasets using concatenation

### Load
- Data stored in pandas DataFrame for analysis
- Ready for visualisation and further processing



### From the Graph you can see

- Between the years 1792 and 1833 the sun spot observations were the lowest on record.  
- That the most sun spots occurd in 1958
- And that we are currently in a solar minimum


### What this means

- If we are able to accuratly predict the suns behaviour we could try to create tools to protect the earth from high solar activty (solar maxium) protecting satalites and the Earths infrustucrute. 
- The taget audenice is astronomers and space weather analysts
 