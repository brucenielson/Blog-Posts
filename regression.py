import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
import numpy as np

# Hard-coded modified data
data = {
    'House ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Square Feet': [1500, 1800, 2000, 1400, 2200, 1600, 2500, 1900, 2100, 2300],
    'Sales Price ($)': [197536, 260942, 282367, 169856, 323267, 199154, 363821, 285262, 280425, 326080]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a table using PrettyTable
table = PrettyTable()
table.field_names = ["House ID", "Square Feet", "Sales Price ($)"]
for i, row in df.iterrows():
    table.add_row(row)

# Print the table
print(table)

# Create a scatter plot
plt.scatter(df['Square Feet'], df['Sales Price ($)'])
plt.xlabel('Square Feet')
plt.ylabel('Sales Price ($)')
plt.title('Relationship between Square Feet and Sales Price')

# Fit a linear regression model
X = df[['Square Feet']]
y = df[['Sales Price ($)']]
reg = LinearRegression().fit(X, y)

# Manually draw the regression line
reg_line_x = [min(df['Square Feet']), max(df['Square Feet'])]
reg_line_y = reg.predict([[min(df['Square Feet'])], [max(df['Square Feet'])]])
plt.plot(reg_line_x, reg_line_y, color='black', linestyle='dotted', label='Linear Regression Line')

# Add a new data point for prediction
new_square_feet = 1700
new_sales_price = reg.predict([[new_square_feet]])[0][0]

# Add the new predicted data point with a bold red "X" marker
plt.scatter(new_square_feet, new_sales_price, color='red', marker='x', s=200, label='New Prediction')

# Draw dotted lines from X and Y axes to the predicted data point
plt.axvline(x=new_square_feet, color='lightgray', linestyle='--')
plt.axhline(y=new_sales_price, color='lightgray', linestyle='--')

# Show the plot with the "X" marker floating on top of the line
plt.legend()
plt.show()

# Print the linear regression coefficients and predicted sales price
print("Linear Regression Coefficients:")
print("Intercept (b0):", reg.intercept_)
print("Slope (b1):", reg.coef_[0])
print(f"Predicted Sales Price for {new_square_feet} Square Feet: ${new_sales_price:.2f}")


# Hard-coded modified data with the number of bedrooms
data = {
    'House ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Square Feet': [1500, 1800, 2000, 1400, 2200, 1600, 2500, 1900, 2100, 2300],
    'Bedrooms': [2, 3, 4, 2, 4, 3, 5, 3, 4, 5],
    'Sales Price ($)': [197536, 260942, 282367, 169856, 323267, 199154, 363821, 285262, 280425, 326080]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a table using PrettyTable
table = PrettyTable()
table.field_names = ["House ID", "Square Feet", "Bedrooms", "Sales Price ($)"]
for i, row in df.iterrows():
    table.add_row(row)

# Print the table
print(table)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Square Feet'], df['Bedrooms'], df['Sales Price ($)'])
ax.set_xlabel('Square Feet')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Sales Price ($)')
ax.set_title('3D Relationship between Square Feet, Bedrooms, and Sales Price')

# Fit a 3D linear regression model
X = df[['Square Feet', 'Bedrooms']]
y = df['Sales Price ($)']
reg = LinearRegression().fit(X, y)

# Create a meshgrid for the plane
square_feet_range = range(min(df['Square Feet']), max(df['Square Feet']), 100)
bedrooms_range = range(min(df['Bedrooms']), max(df['Bedrooms']), 1)
square_feet_mesh, bedrooms_mesh = np.meshgrid(square_feet_range, bedrooms_range)
sales_price_mesh = reg.predict(np.c_[square_feet_mesh.ravel(), bedrooms_mesh.ravel()]).reshape(square_feet_mesh.shape)

# Plot the 3D linear regression plane
ax.plot_surface(square_feet_mesh, bedrooms_mesh, sales_price_mesh, color='cyan', alpha=0.5, label='Linear Regression Plane')

# Show the 3D plot
plt.show()

# Print the 3D linear regression coefficients
print("3D Linear Regression Coefficients:")
print("Intercept (b0):", reg.intercept_)
print("Coefficients (b1, b2):", reg.coef_)
