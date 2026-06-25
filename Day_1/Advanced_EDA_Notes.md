# Day 1: Advanced Exploratory Data Analysis (EDA)

Welcome to Day 1! Today, we will explore the foundation of Data Science: **Exploratory Data Analysis (EDA)**. We will go deep into the concepts but keep the explanations simple, relatable, and easy to understand.

---

## Session 1: Introduction to Data Science & EDA

### What is Data?
Data is simply a collection of facts, observations, or measurements. 
**Examples:**
* Student marks in a class
* Your purchase history on Amazon
* The salary of employees in a company
* Patient medical records in a hospital

### What is Data Science?
Data Science is the process of extracting useful, actionable information from raw data to solve real-world business problems. Think of a Data Scientist as a detective looking for clues in numbers.

**The Data Science Life Cycle:**
1. **Collect Data**: Gather information (e.g., download an Excel file of sales).
2. **Clean Data**: Fix errors (e.g., remove empty rows or correct typos).
3. **Analyze Data (EDA)**: Understand what the data is telling you.
4. **Build Model**: Create a machine learning algorithm to predict the future.
5. **Deploy Model**: Put the model into a real app for people to use.

### What is EDA?
EDA stands for **Exploratory Data Analysis**. 
Think of EDA as getting to know your data before you try to make predictions. You wouldn't buy a car without test-driving it, right? EDA is the test drive for your data!

**Goals of EDA:**
* **Understand the data:** What information do we actually have?
* **Find patterns:** Do sales always spike in December?
* **Detect anomalies (Outliers):** Did someone accidentally type a price of $1,000,000 instead of $10?
* **Identify relationships:** Does a higher education level lead to a higher salary?
* **Generate business insights:** Tell the company exactly what they should do next based on facts.

**Real-world Example:**
Imagine Amazon wants to predict what you will buy next. Before building complex AI, they need to answer:
* Which products sell the most?
* Which cities have the highest number of buyers?
* Are there any missing prices in the database?
* Are there unusually large orders?

This investigation process is called EDA.

---

## Session 2: Understanding a Dataset

Before we can analyze data, we need to load it into our environment and look at it. We use a Python library called **Pandas**, which is essentially "Excel for Python."

### Loading the Data
```python
import pandas as pd

# Load the dataset from a CSV file (Comma Separated Values)
df = pd.read_csv("sales.csv")
```
*(Note: `df` stands for DataFrame, which is just a table of rows and columns).*

### 6 Important Functions Every Data Scientist Uses

1. **`df.head()`**
   * **What it does:** Shows the first 5 rows of the dataset.
   * **Why use it:** To get a quick peek at what the data looks like without scrolling through millions of rows.

2. **`df.tail()`**
   * **What it does:** Shows the last 5 rows.
   * **Why use it:** To check the end of the file (sometimes files have junk data or totals at the bottom).

3. **`df.shape`**
   * **What it does:** Tells you the total number of rows and columns. (e.g., `(1000, 5)` means 1000 rows and 5 columns).
   * **Why use it:** To understand the sheer size of your data.

4. **`df.columns`**
   * **What it does:** Lists all the column headers.
   * **Why use it:** To see what features (variables) you are working with (e.g., 'Age', 'Salary', 'City').

5. **`df.info()`**
   * **What it does:** Shows the data types (text, whole numbers, decimals) and tells you if there are missing values in any column.
   * **Why use it:** To see if Python has misunderstood a number as text, or to quickly spot blank data.

6. **`df.describe()`**
   * **What it does:** Shows basic mathematical statistics for numerical columns (average, minimum, maximum).
   * **Why use it:** To instantly see if numbers make sense. (For example, if the minimum 'Age' is -5, you instantly know there is a mistake in the data!).

**Quick Activity check:** 
Look at `df.info()` and `df.shape` to identify: The number of rows, number of columns, numerical features (numbers), and categorical features (text/categories).

---

## Session 3: Missing Value Analysis

### What are Missing Values?
Sometimes, your table has blanks. 
**Examples:**
* Age = `NaN` (Not a Number)
* Salary = `Null`

### Why do they occur?
* A user simply skipped a question on a signup form.
* The system crashed while saving the data.
* Data got corrupted when transferring from one database to another.

### Finding Missing Values
```python
# This calculates the exact number of missing values in each column
df.isnull().sum()
```

### How to Handle Missing Values
Machine Learning models are like calculators; they break if you feed them blanks. Here are 3 common ways to fix them:

**1. Drop the rows (Delete them)**
* **Use when:** You only have a tiny amount of missing values (like 5 rows out of 10,000), and deleting them won't hurt your analysis.
```python
df.dropna(inplace=True)
```

**2. Replace with the Mean (Average)**
* **What it is:** The mathematical average of the column.
* **The Danger:** If you have one massive number (an outlier), it messes up the average. 
```python
# Fill missing ages with the average age
df["Age"].fillna(df["Age"].mean(), inplace=True)
```

**3. Replace with the Median (Middle value)**
* **What it is:** If you line up all numbers from smallest to largest, the median is the exact middle number.
* **Why it's better:** It is strictly **NOT** affected by outliers. If Bill Gates walks into a bar, the *average* wealth skyrockets, but the *median* wealth stays the same. This is the safest choice for numbers!
```python
# Fill missing ages with the median age
df["Age"].fillna(df["Age"].median(), inplace=True)
```

---

## Session 4: Distribution Analysis

### What is Distribution?
Distribution shows how your data values are spread out across a range. Are most of your customers 20 years old, or 50 years old? 

### The Histogram
We use a visual chart called a **Histogram** to see the spread. It groups numbers into "bins" or ranges.
```python
import matplotlib.pyplot as plt

# Plot a histogram for Age
df["Age"].hist()
plt.show()
```

**Questions to ask yourself when looking at a Histogram:**
* **Is the data symmetric?** Does it look like a perfectly balanced bell curve? (Normal Distribution - this is the holy grail for statistics!).
* **Is the data skewed?** 
  * **Right Skewed:** A long tail pulling to the right. (Example: Salaries. Most people make normal money, but a few billionaires pull the tail far to the right).
  * **Left Skewed:** A long tail pulling to the left. (Example: Retirement age. Most people retire older, very few retire in their 30s).
* **Where are most values concentrated?**

**Business Example:**
If a histogram of "Customer Spending" shows a massive peak between ₹500 and ₹1000, the marketing team should focus their ads on products exactly in that price range!

---

## Session 5: Correlation Analysis

### What is Correlation?
Correlation measures the mathematical relationship between two numerical variables. Basically: Does one number change when the other number changes?

**The Correlation Range (-1 to +1):**
* **+1 (Strong Positive):** As X goes up, Y goes up. (Example: Hours Studied vs. Exam Marks. More study = more marks).
* **0 (No Relationship):** X has absolutely no effect on Y. (Example: Shoe size vs. Exam Marks).
* **-1 (Strong Negative):** As X goes up, Y goes down. (Example: Days Absent vs. Exam Marks. More absences = fewer marks).

### Finding Correlation in Code
```python
# Calculate correlation between all numerical columns
correlation_matrix = df.corr(numeric_only=True)
```

### The Heatmap
Reading a table of correlation decimals is boring. A heatmap is a beautiful, color-coded grid that makes it easy to spot strong relationships instantly.
```python
import seaborn as sns

# Draw a visual heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()
```
**Interpretation:** Look for the darkest red (+1) or darkest blue (-1) squares. These tell you exactly which features strongly affect your target goal (like predicting Sales).

---

## Session 6: Outlier Detection & Business Insights

### What is an Outlier?
An outlier is a data point that is completely, wildly different from the rest of the group.

**Simple Example:**
Look at these ages of kids in a kindergarten class:
`4, 5, 4, 6, 5, 45`
The `45` is the teacher. If we are analyzing *student* ages, the `45` is an outlier!

### Why are Outliers Important?
They can completely ruin your averages and trick your machine learning model into learning the wrong patterns.

### Detection Using a Boxplot
A boxplot is the best visual tool to spot outliers. It draws a box around where most of your data lives. Any tiny dots drawn far outside the "whiskers" of the box are your outliers.
```python
# Plot a boxplot for Salary
sns.boxplot(x=df["Salary"])
plt.show()
```

### Common Methods to Handle Outliers:
* **IQR Method (Interquartile Range):** Uses the 25th and 75th percentiles to define a "normal zone" and flags anything outside it.
* **Z-Score Method:** Uses standard deviations to find data points that are mathematically too far from the average.

### Mini-Assignment: Extracting Business Insights
Data Science isn't just about writing Python code; it's about providing value to the business!

Using a simple sales dataset, try to perform an EDA to answer these 5 questions:
1. **Highest selling category:** What product makes the most money?
2. **Lowest selling category:** What product should we stop selling?
3. **Top revenue city:** Which city is our best market?
4. **Average customer spending:** What is the median order value?
5. **Presence of outliers:** Are there unusually massive orders that might be wholesale buyers or fraud?

*By answering these questions, you've completed your first real-world Exploratory Data Analysis!*
