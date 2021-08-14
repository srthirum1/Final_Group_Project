# Housing Price analysis in CA (Final_Group_Project) 

# Predicting CA Housing Prices  
In this project, we will Predict the average housing prices per SQF for each county in CA. Then we will Visualize the housing prices per county on a Map. 
We will build Machine Learning Models to help investors and homeowners assess the housing prices in California based on Housing transactions. We will build a  supervised neural network machine learning model using the following independent parameters, Zip Code, House Age, SQFT, and Days on Market.



## Data Source
1) MLS Data, https://pro.mlslistings.com/, is deposited at "amazonaws.com" with the name 'big_main.csv". The dataset includes 4344 housing sale transactions in California for the period from 6/2020 till 6/2021. The dataset includes Street_Address	City, Zip_Code, SqFtTotal, Lot_Size, Age, BathsTotal, BedsTotal, BathsFull, BathsHalf, DOM, Year_Sold, Year_List, List_Price, Sale_Price, Listing_Date, Sale_Date, Year_Built. 

2) Additional Data set includes "county_zipcode.csv" which was deposited at "amazonaws.com". This files lists all the Zip Codes to County Name. It is from <a href="https://data.chhs.ca.gov/dataset/ead44d40-fd63-4f9f-950a-3b0111074de8/resource/ec32eece-7474-4488-87f0-6e91cb577458/download/covid19vaccinesbyzipcode_test.csv" target="_blank">California Health and Human Services Open Data Portal</a>

## Questions we Hope to answer:
Motivation, Housing prices are a hot topic, especially During the COVID-19 Pandemic (Leading to increased demand for housing). We strive to build a Machine learning model to Guide investors, Potential Buyers, or Real estate professionals on housing prices per county. 
    1. we Will Build a supervised Neural Machine learning model using  House Age, SQF, Lot size & Days on the Market to predict the median House Price per SQF for the county. 



## Communications Protocols:
o	Members: Trong Quyen, Srividhya Thirumalairajan, Dawit Alaro, Angelica Villanueva & Mikhail Zaatra

##    Segment 1:
• Roles and Responsibilities:
    o	Presentation : Mikhail Zaatra
    o	GitHub: Srividhya Thirumalairajan
    o	Machine Learning Model : Trong Quyen , Angelica Villanueva,  Srividhya Thirumalairajan & Mikhail Zaatra
    o	DataBase: Dawit Alaro  & Angelica Villanueva
    o	DashBoard: N/A 
    
• Project Tools: Collaboration: 
    o	Communication Methods: 
    o	GitHub will be the Main tool for Documents and Code Development. 
    o	- Sharing resources via slack
    o	- using zoom meeting every Thursday as well from 7-9 to work on the project
    
##    Segment 2:

•	Final_data_processing.ipynb:
*   Use google Colab and pyspark, and process this online
*   Combine two data sources mentioned above "big_main.csv" and "county_zipcode.csv"
*   Use "StringIndex" from 'pyspark.ml.feature import' to assign a county name a number
*   Joined two data sets together.  Cleaned up and deleted null values.<br> ![Data Processing Yeah Data is cleaned](tq_folder/images/data_processing_1.png)
*   Export the data to postgresql.  There are three sets of data exported: 'house_data.csv', 'sale_data.csv' and 'final_data.csv'.<br> ![Data Processing data export](tq_folder/images/data_processing_2.png)


•	Regression_Basic.ipynb
* 	Read the 'final_data.csv' from AWS
*   The data has 4225 records. That amount is split 95% for training and 5% for testing.
*   That split means 4013 records are for training and 212 records for testing 
*   The result is impressive, with R squared is 95%.  The model can explain 95% of the price variation.<br> ![Regression Basic MSE and R squared](tq_folder/images/Regression_Basic_1.png)
*   The coefficients for the colums "County_Index|SqFtTotal|Lot_Size|Age|BathsTotal|BedsTotal|BathsFull|BathsHalf|DOM|Year_Sold|List_Price" is below:<br> ![Regression Basic Model Coefficients](tq_folder/images/Regression_Basic_2.png)
*   With this, one can build an estimate calculator.
*   With the coefficients above, we applied the coefficient to all the sales, and created a new column "Predicted Value".
*   "final_data.csv" with an extra column of "Predicted Value" is saved as "final_prediction_all.csv". This "final_prediction_all" will be used for mapping and visualization.

##    Segment 3:

### Final_Regression_NN.ipynb
*   This file is considered "final" because this file is a merge of Regression and Neural Network models together in one file
*   The purpose is to force two models to accept the same number of X_train, y_train, X_test, and y_test data points
*   This final file is even further broken down into 4 models: Regression with List_Price, Regression without List_Price, Neural Network with List_Price, and Neural_Network without List_Price.

![Final Descriptive](Final_Project/images/1_Final_descriptive.png)

####1.  Model 1: Regression with List_Price
*   We have a list of coefficients. The List_Price is highly correlated with the Sale_Price, the model independent variable.
<br>![model 1 coefficients](Final_Project/images/2_model1_coefficient.png)
*   We have some metrics for model 1.  The R2_squared is 99%
<br>![model 1 metrics](Final_Project/images/3_model1_metrics.png)
*   Here is an example how model 1 predicts:
<br>![model 1 Sale Price examples](Final_Project/images/4_model1_saleprice_examples.png)
*   Here is the plot of the model 1 residuals
<br>![model 1 residuals plot](Final_Project/images/5_model1_residuals_plot.png)
*   The prediction price plot can tells how closely model 1 predicts its Sale_Price
<br>![model 1 Sale Price prediction plot](Final_Project/images/6_model1_prediction_plot.png)

####2.  Model 2: Regression without List_Price
*   We have a list of coefficients for model 2.  There is no coefficient for List_Price
<br>![model 2 coefficients](Final_Project/images/7_model2_coefficient.png)
*   Notice how the r2_square drops significantly to 65%. This means that model 2 can only predict 65.9% of the sale_price accurately.
<br>![model 2 metrics](Final_Project/images/8_model2_metrics.png)
*   Look at how well apart model 2 predicts Sale_Price
<br>![model 2 Sale Price examples](Final_Project/images/9_model2_saleprice_examples.png)
*   Take a peak at the model 2 residuals plot:
<br>![model 2 residual plot](Final_Project/images/10_model2_residuals_plot.png)
*   As expected, model 2 prediction is not as accurate as model 1
<br>![model 2 sale price prediction plot](Final_Project/images/11_model2_prediction_plot.png)

####3.  Model 3: Neural Network with List_Price




•	Roles and Responsibilities:
*   Presentation : Mikhail Zaatra
*   GitHub: Srividhya Thirumalairajan
*   Machine Learning Model : Trong Quyen , Srividhya Thirumalairajan & Mikhail Zaatra
*   DataBase: Dawit Alaro  & Angelica Villanueva
*   DashBoard: N/A 

