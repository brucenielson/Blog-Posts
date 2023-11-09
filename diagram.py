from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression

# # Create a Venn diagram
# venn = venn2(subsets=(2, 0, 1), set_labels=('', ''))  # Set empty labels
#
# # Set the set labels outside the diagram
# venn.get_label_by_id('10').set_text('Artificial Intelligence')
# venn.get_label_by_id('01').set_text('Machine Learning')
# venn.get_label_by_id('11').set_text('')
#
#
# # Display the diagram
# plt.title("AI and ML Relationship")
# plt.show()

from matplotlib_venn import venn2

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
plt.plot(X, reg.predict(X), color='red', linewidth=2, label='Linear Regression Line')
plt.legend()

# Show the plot
plt.show()

# Print the linear regression coefficients
print("Linear Regression Coefficients:")
print("Intercept (b0):", reg.intercept_)
print("Slope (b1):", reg.coef_[0])
