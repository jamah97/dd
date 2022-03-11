#core Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd 
import numpy as np 
import time,json

# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import altair as alt 
st.set_option('deprecation.showPyplotGlobalUse', False)

# ML pkgs 
import pickle


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
  # Load Our Dataset
df = pd.read_csv("/content/diabetes.csv")
	  # feature selection 
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
		# feature scaling 
#sc = StandardScaler()
#X = sc.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.15,random_state = 5)


#model Logistic Regression
reg = LogisticRegression()
reg.fit(X_train, y_train)   
y_pred=reg.predict(X_test)
#print('Logistic Regression accuracy score:', accuracy_score(y_test,y_pred)*100)


file = open('DD_model.pkl', 'wb')
pickle.dump(reg, file)


model = open('DD_model.pkl', 'rb')
DT=pickle.load(model) # our model


def predict_chance(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    prediction=DT.predict_proba([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]) #predictions using our model
    return prediction 

def main():

	activities = ["About", "EDA","ML Model",]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice =='About':
		st.subheader("About section")

		st.write(""" App created by Hassan Jama. The app has 2 other section EDA and ML model you will be able to check them out on the side bar.""")
		st.write(""" About dataset: 
		This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. """)
		
		st.write("""Content: 
		The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.""")

		st.write("""Datasource : 
		https://www.kaggle.com/uciml/pima-indians-diabetes-database """)
		st.write(""" Dataset biases/shortcomings:
		There is a regional bias as all of the participants are of Pima Indian heritage. Also all of the participants are female and the dataset consist of only 768 participants. Being aware of these biases is key to understanding predictions made on the ML section of this app, as the training and test datasets used to build the model have these shortcomings.""")

		st.write("""Algorithm used:
		The algroithm used on the ML section of the app is an Logistic Regression with an accuracy score of 81.3%. To learn more about decision trees algrothim click on this link https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html""")



	elif choice == 'EDA':
		st.subheader("Exploratory Data Analysis & Data Visualization")

	# Load Our Dataset
		df = pd.read_csv("/content/diabetes.csv")
 
		group_names = ['20\'s', '30\'s', '40\'s', '50\'s', '60\'s', '70\'s']
		df['AgeBD'] = pd.cut(df['Age'], bins=[20,29,39,49,59,69,79], labels=group_names, include_lowest=True)

		group_names_bmi = ['Underweight', 'Healthy', 'Overweight', 'Obese1','Obese2', 'Obese3']
		df['BMIBD'] = pd.cut(df['BMI'], bins=[0,18.5, 24.9, 29.9, 34.9, 39.9, 50 ], labels=group_names_bmi,include_lowest=True)
		bmibd = df.BMIBD.value_counts().to_frame()
 
		if st.checkbox("Show DataSet"):
			st.dataframe(df)
 
		if st.checkbox("Show missing Data"):
			st.write(df.isnull().sum())

		if st.checkbox("Show Shape"):
			st.write(df.shape)

		if st.checkbox("Show Columns"):
			all_columns = df.columns.to_list()
			st.write(all_columns)

		if st.checkbox("Summary"):
			st.write(df.describe())

		if st.checkbox("Selected Columns"):
			selected_columns = st.multiselect("Select Columns", df.columns.to_list())
			new_df = df[selected_columns]
			st.dataframe(new_df)
	
		if st.checkbox("Show Value Count of Age groups"):
			st.dataframe(df['AgeBD'].value_counts())
			st.write(sns.barplot(group_names, df["AgeBD"].value_counts()))
			st.pyplot()
	
		if st.checkbox("Show Value Count of BMI groups"):
			st.dataframe(df['BMIBD'].value_counts())
			st.write(sns.barplot(bmibd.index, df["BMIBD"].value_counts()))
			st.pyplot()

		if st.checkbox("Show Value Count of Target Variable"):
			st.write(df['Outcome'].value_counts())


		if st.checkbox("Correlation Plot(Seaborn)"):
			st.write(sns.heatmap(df.corr(),annot=True, linewidths=2, cmap = "YlGnBu"))
			st.pyplot()
  
		if st.checkbox('Scatterplot'):
			all_columns_names1 = df.columns.tolist()
			columnsx = st.selectbox("Select X Column",all_columns_names1)
			columnsy = st.selectbox("Select Y Column",all_columns_names1)
			st.write(sns.scatterplot(x = columnsx, y = columnsy, hue = 'Outcome', data = df, alpha = 1))
			#st.write(sns.scatterplot(data=df, x=columnsx, y=columnsy))
			st.pyplot()


		if st.checkbox("Pie Plot"):
			all_columns = df.columns.to_list()
			column_to_plot = st.selectbox("Select 1 Column",all_columns)
			pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()



		all_columns_names = df.columns.tolist()
		type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
		selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

		if st.button("Generate Plot"):
			st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
			if type_of_plot == 'area':
				cust_data = df[selected_columns_names]
				st.area_chart(cust_data)

			elif type_of_plot == 'bar':
				cust_data = df[selected_columns_names]
				st.bar_chart(cust_data)

			elif type_of_plot == 'line':
				cust_data = df[selected_columns_names]
				st.line_chart(cust_data)


				# Custom Plot 
			elif type_of_plot:
				cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
				st.write(cust_plot)
				st.pyplot()
		
	elif choice == 'ML Model':
		st.subheader("ML Model Diabetes")
		html_temp="""
        <div>
        <h2>Diabetes Prediction ML app</h2>
        </div>
        """
		st.markdown(html_temp,unsafe_allow_html=True) #a simple html 
		Pregnancies=st.number_input("Pregnancies")
		Glucose=st.number_input("Glucose")
		BloodPressure=st.number_input("Blood Pressure")
		SkinThickness=st.number_input("Skin Thickness")
		Insulin=st.number_input("Insulin")
		BMI=st.number_input("BMI")
		DiabetesPedigreeFunction=st.number_input("Diabetes Pedigree Function")
		Age=st.number_input("Age")
	

     #giving inputs as used in building the model
		result=""
		if st.button("Predict"):
				result=predict_chance(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
				#if result == 1:
					#st.success('You likely have diabetes', )
				#else:
					 #st.success('You likely do not have diabetes')
		st.success("Your Result from the inputs above is{}".format(result))


 
if __name__ == '__main__':
	main()
