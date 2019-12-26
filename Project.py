import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import linear_model
from scipy import stats

import datetime
import calendar
import csv
import warnings

warnings.filterwarnings('ignore')

## method to accept company name from user
def accept_company_name():
    found_company = False
    ## run until a company is selected
    while found_company == False:
        with open('/Users/dastan/Desktop/StocksProject/companylist.csv') as f:
            company_reader = csv.reader(f)
            company_name = input(
                "Please enter the company name to analyse the stock data: \n")
            for row in company_reader:
                if company_name in row[1].lower():
                    print('Company name: ', row[1])
                    print('Ticker Code: ', row[0])
                    found_company = True
        if not found_company:
            print("\nSorry we could not find the company with the entered name.")

## method to accept company tickr code from user
def accept_tickr_code():
    tickr_code = None
    found_tickr = False

    while not found_tickr:
        print("\nPlease select company tickr code from the list above. ")
        tickr_code = input().upper()
        ticker_list = []
        with open('/Users/dastan/Desktop/StocksProject/companylist.csv') as f:
            for row in f:
                ticker_list.append(row.split(',')[0])
        if tickr_code in ticker_list:
            print("Entered company tickr code is correct")
            found_tickr = True
        else:
            print("Please enter the correct company tickr code !")

    return tickr_code

## method to accept start and end dates
def accept_start_end_dates():
    start_flag = False
    while not start_flag:
        start_input = input('\nPlease enter the Start Date in YYYY-MM-DD format:')
        start_year, start_month, start_day = map(int, start_input.split('-'))
        start_flag = validate_date(start_year, start_month, start_day)
        if not start_flag:
            print("Please enter a valid date in the specied format")
        else:
            print("Thank you. The start date entered is Valid")

    end_flag = False
    while not end_flag:
        end_input = input(
            '\n Please enter the end Date to see the descriptive analysis of the company in YYYY-MM-DD format:')
        end_year, end_month, end_day = map(int, end_input.split('-'))
        end_flag = validate_date(end_year, end_month, end_day)
        if not end_flag:
            print("Please enter a valid date in the specied format")
        else:
            if end_input < start_input or end_input == start_input :
                print("Invalid end date. End date should be greater than start date !")
                end_flag = False

    return start_year, start_month, start_day, end_year, end_month, end_day

## method to check if the user has enetered a valid date
def validate_date(year, month, day):
    date_time = False
    try:
        datetime.datetime(year, month, day)
        date_time = True
    except Exception:
        date_time = False
    return date_time

def trend_line(stock_df):
    data0 = stock_df.copy()
    data0['Date'] = ((stock_df.index.date - stock_df.index.date.min())).astype('timedelta64[D]')
    data0['Date'] = data0['Date'].dt.days + 1

    #high trend line
    data1 = data0.copy()

    while len(data1) > 3:

        reg = stats.linregress(
            x=data1['Date'],
            y=data1['High'],
        )
        data1 = data1.loc[data1['High'] > reg[0] * data1['Date'] + reg[1]]

    reg = stats.linregress(
        x=data1['Date'],
        y=data1['High'],
    )
    data0['high_trend'] = reg[0] * data0['Date'] + reg[1]

    # Low Trend line
    data1 = data0.copy()

    while len(data1) > 3:

        reg = stats.linregress(
            x=data1['Date'],
            y=data1['Low'],
        )
        data1 = data1.loc[data1['Low'] < reg[0] * data1['Date'] + reg[1]]

    reg = stats.linregress(
        x=data1['Date'],
        y=data1['Low'],
    )
    data0['Low_trend'] = reg[0] * data0['Date'] + reg[1]

    data0['Close'].plot()
    data0['high_trend'].plot()
    data0['Low_trend'].plot()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.suptitle(tickr_code+' Stock Price')
    plt.show()

## method to plot the moving average graph
def moving_average(stock_df):
    n_value= input("Enter the n value for Moving Average: ")
    stock_df['Moving_Average'] = stock_df['Close'].rolling(window=int(n_value), min_periods=0). mean()
    ma_plot = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ma_plot.plot(stock_df.index, stock_df['Close'], label = "Close")
    ma_plot.plot(stock_df.index, stock_df['Moving_Average'], label = "Moving Average")
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.suptitle(tickr_code+' Stock Price')
    plt.show()

##  method to predict prices using linear regression
def predict_price(ordinal_dates, price_list, x):
    # defining the linear regression model
    reg_model = linear_model.LinearRegression()
    ordinal_dates = np.reshape(ordinal_dates, (len(ordinal_dates), 1))  # converting to matrix of n X 1
    price_list = np.reshape(price_list, (len(price_list), 1))
    reg_model.fit(ordinal_dates, price_list)  # fitting the data points in the model
    price = reg_model.predict([[x]])
    return price[0][0], reg_model.coef_[0][0], reg_model.intercept_[0]


# Start of the program
print("\n *** Welcome to Business Analytics Project! ***\n")

run = True
while run:
    ## accepting company name displaying tickr codes
    accept_company_name()

    ## accepting the TICKR code from the user
    tickr_code = accept_tickr_code()
    
    ## accepting and validating start and end dates 
    start_year, start_month, start_day, end_year, end_month, end_day = accept_start_end_dates()

    # Getting the data for the entered company online and storing in df
    start_date = datetime.datetime(start_year, start_month, start_day)
    end_date = datetime.datetime(end_year, end_month, end_day)
    ## using yahoo service to fetch company stock details
    stock_df = web.DataReader(tickr_code, "yahoo", start_date, end_date)
    ## saving data frame into a csv file
    stock_df.to_csv('company_details.csv')

    # Showing the descriptive analysis of the entered company
    print("\nDescriptive analytics of the entered company")
    ## general discription of the dataframe
    print(stock_df.describe())
    closing_val_variation = stats.variation(stock_df['Close'], axis=0)
    print("\nVariation=", closing_val_variation)

    # Graphically visualising the data and prediction for any company

    while True:
        print("\n Menu:  ")
        print("1. Trend Line")
        print("2. Raw Time-Series and Moving Averages")
        print("3. Raw Time-Series and Weighted Moving Averages")
        print("4. MACD")
        print("5. Predict the stock price")
        print("6. Quit for current company")
        user_choice = input("\nPlease choose an option between 1 and 6: ")

        if user_choice == "1":
            trend_line(stock_df)

        elif user_choice == "2":
            moving_average(stock_df)

        elif user_choice == "3":
            print("Enter the n value for Weighted Moving Average")
            n_value = input()
            stock_df['fw_exp_wm_avg'] = stock_df['Close'].ewm(span=int(n_value)).mean()
            wma_plot = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
            wma_plot.plot(stock_df.index, stock_df['Close'], label='Close')
            wma_plot.plot(stock_df.index, stock_df['fw_exp_wm_avg'], label='WMA')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.suptitle(tickr_code+' Stock Price')
            plt.show()

        elif user_choice == "4":
            x = 26
            y = 12
            stock_df['26 ema'] = stock_df['Close'].ewm(span=int(x)).mean()
            stock_df['12 ema'] = stock_df['Close'].ewm(span=int(y)).mean()
            stock_df['MACD'] = (stock_df['12 ema'] - stock_df['26 ema'])
            macd_plot = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
            macd_plot.plot(stock_df.index, stock_df['Close'], label='Close')
            macd_plot.plot(stock_df.index, stock_df['MACD'], label='MACD')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.suptitle(tickr_code+' Stock Price')
            plt.show()

        elif user_choice == "5":
            predict_date_flag = False
            while not predict_date_flag:
                query_date = input(
                    "Enter the date for which you want to predict the stock price in yyyy-mm-dd format:")
                pred_year, pred_month, pred_day = map(int, query_date.split('-'))
                predict_date_flag = validate_date(start_year, start_month, start_day)

                if not predict_date_flag:
                    print("Enter the correct date")

                else:
                    # calculate the ordinal value of the prediction date
                    to_predict_date = datetime.date(year=pred_year, month=pred_month, day=pred_day).toordinal()
                    days = []
                    months = []
                    years = []
                    prices = []
                    ordinal_days = []

                    with open('/Users/dastan/Desktop/StocksProject/company_details.csv', 'r') as company_file:
                        company_file_reader = csv.reader(company_file)
                        next(company_file_reader)  # skipping column names
                        for row in company_file_reader:
                            d_values = row[0].split('-')
                            days.append(int(d_values[2]))
                            months.append(int(d_values[1]))
                            years.append(int(d_values[0]))
                            prices.append(float(row[4]))

                    ## storing all the ordinal values of dates into an array
                    for index in range(len(days)):
                        day_val = datetime.date(year=years[index], month=months[index], day=days[index])
                        day_val = day_val.toordinal()
                        ordinal_days.append(day_val)

                    predicted_price, coeff, constant = predict_price(ordinal_days, prices, to_predict_date)
                    print("The stock predicted price for "+str(pred_year) + "-" +
                          str(pred_month)+"-"+str(pred_day)+" is: $", str(predicted_price))
                    print("The regression coefficient is ", str(coeff))

        elif user_choice == "6":
            break
        else:
            print("Incorrect entry. Please try again !")

    print("Do you want to check for another company 1. Yes  2. No")
    user_choice2 = input()
    if int(user_choice2) == 1:
        run = True
    else:
        run = False
