import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objs as go

class StockAnalysisApp:
    def __init__(self):
        # Load pre-trained model and data
        self.model = joblib.load('/home/hbs/Documents/Meenakshi/stock_prediction_model1.pkl')
        self.df = pd.read_csv('/home/hbs/Downloads/df1.csv')
    
    def correlation_analysis(self):
        st.title('Stock Correlation Analysis')
        
        # Stock-to-Stock Correlation
        st.subheader('Stock-to-Stock Correlation Matrix')
        returns_df = self.df.pivot(index='Date', columns='Symbol', values='Return')
        corr_matrix = returns_df.corr()
        
        # Interactive Heatmap
        fig = px.imshow(corr_matrix, 
                        labels=dict(x="Stocks", y="Stocks", color="Correlation"),
                        x=corr_matrix.columns, 
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        
        # Sector Correlation
        st.subheader('Sector Correlation Analysis')
        sector_returns = self.df.groupby(['Date', 'Sector'])['Return'].mean().unstack()
        sector_corr = sector_returns.corr()
        
        sector_fig = px.imshow(sector_corr, 
                                labels=dict(x="Sectors", y="Sectors", color="Correlation"),
                                x=sector_corr.columns, 
                                y=sector_corr.columns,
                                color_continuous_scale='RdBu_r')
        st.plotly_chart(sector_fig)
    
    def stock_prediction(self):
        st.title('Stock Performance Prediction')
        
        # Select features for prediction
        selected_features = st.multiselect(
            'Select Features for Prediction', 
            ['Return', 'Volatility', 'RSI', 'SMA_20', 'SMA_50', 'Log_Return', 'EMA_20', 'EMA_50',
       'MACD_Line', 'Signal_Line', 'Middle_Band', 'STD', 'Upper_Band',
       'Lower_Band', 'ROC', 'Momentum']
        )
        
        # Input features
        input_data = {}
        for feature in selected_features:
            input_data[feature] = st.number_input(f'Enter {feature}', value=0.0)
        
        # Predict button
        if st.button('Predict Stock Performance'):
            if len(input_data) == len(selected_features):
                input_array = np.array([input_data[feature] for feature in selected_features])
                prediction = self.model.predict(input_array.reshape(1, -1))
                proba = self.model.predict_proba(input_array.reshape(1, -1))
                
                st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
                st.write(f"Probability: {proba[0][prediction[0]]:.2%}")
            else:
                st.error("Please fill in all selected features")
    
    def run(self):
        st.sidebar.title('Stock Analysis Dashboard')
        app_mode = st.sidebar.selectbox(
            'Choose Analysis Mode',
            ['Correlation Analysis', 'Stock Prediction']
        )
        
        if app_mode == 'Correlation Analysis':
            self.correlation_analysis()
        else:
            self.stock_prediction()

# Run the app
if __name__ == '__main__':
    app = StockAnalysisApp()
    app.run()