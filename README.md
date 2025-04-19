⛏️ Gold Mining Recovery Analysis

This project analyzes and predicts gold recovery rates in mining operations. Using real-world data, the notebook builds and evaluates regression models, compares recovery stages, and identifies process inefficiencies.

📚 Table of Contents
About the Project
Installation
Usage
Project Structure
Technologies Used
Results & Insights
Screenshots
Contributing
License

📌 About the Project
This notebook walks through:

Exploratory Data Analysis (EDA) of gold recovery stages

Data cleaning and engineering

Machine learning using Linear Regression and Random Forest

Model validation with RMSE

Comparative analysis of full and rough recovery processes

🛠 Installation
Clone the repository or download the .ipynb file

Install the required libraries:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Launch Jupyter:

bash
Copy
Edit
jupyter notebook
🚀 Usage
Open the file Gold Mining Recovery Analysis.ipynb and run all cells sequentially. It will:

Load and process multiple gold recovery datasets

Visualize recovery performance and flow

Train and evaluate predictive models

Present findings on which recovery methods are more effective

📁 Project Structure
bash
Copy
Edit
Gold Mining Recovery Analysis.ipynb   # Main analysis notebook
README.md                             # Project documentation
images_goldmining/                    # Visualizations and plots
⚙️ Technologies Used
Python 3.8+

Pandas

NumPy

Scikit-learn

Seaborn

Matplotlib

Jupyter Notebook

📊 Results & Insights
After importing three raw data files from a gold mining company work began to build a model to predict rougher concentrate (recovery rougher.output.recovery) and final concentrate recovery (final.output.recovery). Recovery was verified to be calculated correctly.  In the training set the MAE between calculations and the feature values was so miniscule (9.3e-15). Some features were noted to not be available in the test set (35 columns).  These column values were merge in from the full raw data file.  

Some trends were noticed regarding elements throughout the gold purification process.  Golds highest concentration was at the end of the process; indicating a successful purification process.  Silver's highest concentration during the purification process is the product output from the second cleaning (primary_cleaner.output.tail_ag).  The presence of both lead and silver through the purification process was minority volumes. The average feed size is exactly the same according to the phase of gold purification between the two dataframes.  This is an indication that the machine learning model will be well trained since similiar data is present in both dataframes. 

Discovery was made that the best model to use was a Random Forest Regressor.  In testing the training model this achieved the low sMAPE value of 10.67%. The variables were then redefined to match the test set dataframe.  The optimization random forest regressor for 'roughter.output.recovery' and 'final.output.recovery' resulted in a sMAPE value of 15.24%. Both of these values were used to calculate the final sMAPE value of 0.141%. This model proficiently predicts gold recovery, helping to optimize production and eliminate unprofitable parameters. This effort paves the way for improved operational efficiency.

📸 Screenshots
### 🧪 Null Value Heatmap  
![Null Heatmap](images/goldmining_image_1.png)

### 🔍 Concentration Over Time  
![Concentration](images/goldmining_image_2.png)

### 📈 Rough vs Final Recovery  
![Recovery Comparison](images/goldmining_image_3.png)

### 🔢 Correlation Matrix  
![Correlation](images/goldmining_image_4.png)

### 📉 RMSE Comparison  
![RMSE Plot](images/goldmining_image_5.png)

### 🔁 Recovery Flow Diagram  
![Flow](images/goldmining_image_6.png)

### 📉 Feature Distribution  
![Distribution](images/goldmining_image_7.png)

### ⚙️ Feature Importance  
![Feature Importance](images/goldmining_image_8.png)


🤝 Contributing
Feel free to fork this project and explore:

Time-series models for recovery prediction

Ensemble methods or XGBoost

Deployment of the best model as a web app

🪪 License
This project is open-source and available under the MIT License.

