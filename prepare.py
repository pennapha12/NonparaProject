import pandas as pd
import numpy as np
from scipy.stats import stats
from scipy.stats import spearmanr, pearsonr
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
from scipy.stats import median_test
from scipy.stats import friedmanchisquare
from scipy.stats import mannwhitneyu

#-----------------------------------------------------------------------------------------------------------------------
'''Import data'''
#Saraly
file_path = r'C:\Nonpara_Project\Software Engineer Salaries.csv'
data = pd.read_csv(file_path)
print(data.head())
data.columns

#-----------------------------------------------------------------------------------------------------------------------
##clean data Location 

# Create new columns 'Country' and 'City' by splitting the 'Location' column
data[['City', 'state']] = data['Location'].str.split(',', expand=True)
# แสดงผลคอลัมน์ 'City', 'state', และ 'Location'
print(data[['Location','City', 'state']].head())

#แปลงใช้ Location เป็นตัวพิมเล็กหมด เพื่อให้สามารถดึงข้อมูลได้อย่างครบถ้วน
data['Location']=data['Location'].str.lower()
#สร้างคอลัมน์ใหม่ 'Remote' ที่ตรวจสอบว่ามี 'Remote' ใน 'Location' หรือไม่ 
data['Remote'] = data['Location'].str.contains('remote').fillna(False)
data['Remote'] = data['Remote'].replace({True: 'Yes', False: 'No'})
print(data[['Location','Remote']].head())



#------------------------------------------------------------------------------------------------------------------
##clean data salary
def extract_salary_range(salary_str):
    """Extracts the lower and upper bounds of the salary range from a string.
    Args:
        salary_str: The salary string (e.g., '$68K - $94K (Glassdoor est.)').
    Returns:
        A tuple containing the lower and upper salary bounds (as integers), or None if
        the string cannot be parsed.
    """
    # ตรวจสอบว่าค่าที่รับเข้ามาเป็นสตริงหรือไม่
    if isinstance(salary_str, str):
        # ลบอักขระที่ไม่จำเป็น
        salary_str = salary_str.replace('(', '').replace(')', '').replace('$', '').replace('K', '')
        # แยกสตริงตามเครื่องหมายขีด
        parts = salary_str.split('-')
        if len(parts) == 2:  # ถ้ามีช่วงเงินเดือน
            try:
                lower = int(parts[0].strip())
                upper = int(parts[1].strip().split()[0])  # จัดการกับข้อความส่วนเกินหลังขีดบน
                return lower, upper
            except ValueError:
                return None
        elif len(parts) == 1:  # ถ้ามีแค่ค่าเดียว
            try:
                value = int(parts[0].strip())
                return value, value  # กำหนดให้ค่าขีดบนและขีดล่างเท่ากัน
            except ValueError:
                return None
    return None  # ถ้าค่าไม่สามารถถูกแปลงได้

#ใช้งานฟังก์ชันกับคอลัมน์ Salary
data[['LowerSalary', 'UpperSalary']] = data['Salary'].apply(lambda x: pd.Series(extract_salary_range(x)))
#สร้าง columns Mid_Saraly เพื่อใช้เป็นตัวแทนของข้อมูลเงินเดือนที่กรอกค่ามาให้เป็นช่วง
data['Mid_Salary'] = [(lower + upper) / 2 for lower, upper in zip(data['LowerSalary'], data['UpperSalary'])]
data.sort_values(by='Mid_Salary',ascending=False).head()

#----------------------------------------------------------------------------------------------------------------------------
##clean data Job Title 
#ทำการ lower txt ใน Job Title เพื่อทำสามารถดึงข้อมูลได้อย่างครบถ้วน 
data['Job Title'] = data['Job Title'].str.lower()
# สร้างคอลัมน์ 'Group_job' โดยกำหนดค่าเริ่มต้นเป็น 'Other'
data['Group_job'] = 'Other'
# กำหนดเงื่อนไขสำหรับการจัดกลุ่ม
data.loc[data['Job Title'].str.contains('dev',case=False), 'Group_job'] = 'Software_Dev'
data.loc[data['Job Title'].str.contains('software', case=False), 'Group_job'] = 'Software_En'
data.loc[data['Job Title'].str.contains('cloud', case=False), 'Group_job'] = 'Software_clound'
print(data[['Job Title','Group_job']].head())

# นับจำนวนค่าที่ไม่ซ้ำในคอลัมน์ 'Group_job'
unique_counts = data['Group_job'].value_counts().sort_values(ascending=False)
print(unique_counts)

##----------------
'''
import matplotlib as plt
# สร้างกราฟแท่ง
plt.figure(figsize=(10, 6))
unique_counts.plot(kind='bar', color='skyblue')
# ตั้งค่าชื่อและป้าย
plt.title('Count of Unique Values in Group_job', fontsize=16)
plt.xlabel('Group Job', fontsize=14)
plt.ylabel('Count', fontsize=14)
# แสดงค่าตัวเลขบนแท่งกราฟ
for index, value in enumerate(unique_counts):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##--------------------
'''
#----------------------------------------------------------------------------------------

# คำนวณค่าเฉลี่ยของ Mid_Salary ตาม Group_job
group_mean = data.groupby('Group_job')['Mid_Salary'].transform('mean')
group_mean.value_counts()

# แทนค่าว่างใน Mid_Salary ด้วยค่าเฉลี่ยของกลุ่ม
data['Mid_Salary'].fillna(group_mean, inplace=True)
print(data.head())

# สร้างช่วงเงินเดือนที่กำหนดเอง
bins = [-float('inf'), 50, 100, 150, 200,250,300,float('inf')]  # ขีดจำกัดของแต่ละช่วง
labels = ['<50', '50-100', '101-150', '151-200','201-250','251-300','>300']  # ป้ายกำกับแต่ละช่วง
# แบ่งข้อมูลเงินเดือน (Mid_Salary) ออกเป็นช่วง
data['Salary_Range'] = pd.cut(data['Mid_Salary'], bins=bins, labels=labels, right=True)
data['Salary_Range'].value_counts()


#-------------------------------------------------------------------------------------------------------
##แทนค่าว่าใน Company ด้วย Unknow
data['Company'].fillna('Unknow',inplace=True)
print(data['Company'])

#--------------------------------------------------------------------------------------------------------
#แทนค่าว่างในคะแนนบริษัทด้วยคะแนนเฉลี่ย
data['Company Score'].fillna(data['Company Score'].mean(),inplace=True)
print(data['Company Score'])

# แบ่งกลุ่มคะแนนบริษัทเป็น 3 ช่วง
# กำหนดช่วงคะแนน (เพิ่มช่วงบนเพื่อครอบคลุมคะแนนสูง)
bins_score = [2,3.5, 4.5, float('inf')]  # ช่วงคะแนนต่ำ, กลาง, สูง
labels_score = ['ต่ำ(0-2]', 'กลาง(>2-3.5]', 'สูง(>3.5-5]']
# ใช้ pd.cut() เพื่อแบ่งกลุ่มคะแนน
data['Company_Score_Group'] = pd.cut(data['Company Score'], bins=bins_score, labels=labels_score, right=False)
# แสดงจำนวนคะแนนในแต่ละกลุ่ม
print(data['Company_Score_Group'].value_counts())

#--------------------------------------------------------------------------------------------------------------------
##Prepare data
# แยกข้อมูลเงินเดือนตามกลุ่มงาน
Sa_Software_En = data[data['Group_job'] == 'Software_En']['Mid_Salary']
Sa_Software_Dev = data[data['Group_job'] == 'Software_Dev']['Mid_Salary']
Sa_Software_cloud = data[data['Group_job'] == 'Software_clound']['Mid_Salary']
Sa_Other = data[data['Group_job'] == 'Other']['Mid_Salary']
# บันทึกเป็นไฟล์ CSV
Sa_Software_En.to_csv('en_salary.csv', index=False)
Sa_Software_Dev.to_csv('dev_salary.csv',index=False)
Sa_Software_cloud.to_csv('cloud_salary.csv',index=False)
Sa_Other.to_csv('other_salary.csv',index=False)


#--------------------------------------------------------------------------------------------------

# ทำการทดสอบ Kruskal-Wallis
stat, p_value = kruskal(Sa_Software_En, Sa_Software_Dev, Sa_Software_cloud, Sa_Other)
# แสดงผลลัพธ์
print(f"Kruskal-Wallis H-statistic: {stat}")
print(f"P-value: {p_value}")

#---------------------------------------------------------------------------------------------------

from scipy.stats import spearmanr, pearsonr
# คำนวณ Spearman correlation
spearman_corr, spearman_p_value = spearmanr(data['Mid_Salary'], data['Company Score'])
# แสดงผลลัพธ์
print(f"Spearman Correlation Coefficient: {spearman_corr}")
print(f"P-value: {spearman_p_value}")

# คำนวณ Pearson correlation
pearson_corr, pearson_p_value = pearsonr(data['Mid_Salary'], data['Company Score'])
print(f"Pearson Correlation Coefficient: {pearson_corr}")
print(f"Pearson P-value: {pearson_p_value}")

#-------------------------------------------------------------------
mid_salary=data['Mid_Salary']
mid_salary

company_score=data['Company Score']
company_score

mid_salary.to_csv('mid_salary.csv',index=False)
company_score.to_csv('company_score.csv',index=False)


#---------------------------------------------------------------------
# สร้างตารางไขว้ (contingency table) ระหว่าง Salary_Range และ Company_Score_Group
contingency_table_sa_co = pd.crosstab(data['Salary_Range'], data['Company_Score_Group'])
contingency_table_sa_co.to_csv('salary_score_range.csv',index=False)

# ทำการทดสอบ Chi-Square
chi2, p_value, dof, expected = chi2_contingency(contingency_table_sa_co)

# แสดงผลลัพธ์
print("Expected frequencies:")
print(expected)
print(f"Degrees of freedom: {dof}")
print(f"Chi-Square statistic: {chi2}")
print(f"P-value: {p_value}")

#-----------------------------------------------------------------------------------------------

data['Group_job'].value_counts()
data['Remote'].value_counts()
Software_En=data[data['Group_job']=='Software_En']
Software_Dev=data[data['Group_job']=='Software_Dev']
Software_clound=data[data['Group_job']=='Software_clound']

# Separate Remote (Yes) and Onsite (No) salaries
en_remote_salaries = Software_En[Software_En['Remote'] == 'Yes']['Mid_Salary']
en_onsite_salaries = Software_En[Software_En['Remote'] == 'No']['Mid_Salary']

en_onsite_salaries.to_csv('en_onsite_salary.csv',index=False)
en_remote_salaries.to_csv('en_remote_salary.csv',index=False)
#--------------------------------

from scipy.stats import median_test

# Perform Median Test
if not en_remote_salaries.empty and not en_onsite_salaries.empty:
    stat_en, p_value_en, _, _ = median_test(en_remote_salaries, en_onsite_salaries)
    print(f"Median Test statistic: {stat_en}")
    print(f"P-value (Median Test): {p_value_en}")

Software_Dev[Software_Dev['Remote']=='Yes']['Mid_Salary'].median()
Software_Dev[Software_Dev['Remote']=='No']['Mid_Salary'].median()

# Separate Remote (Yes) and Onsite (No) salaries
Dev_remote_salaries = Software_Dev[Software_Dev['Remote'] == 'Yes']['Mid_Salary']
Dev_onsite_salaries = Software_Dev[Software_Dev['Remote'] == 'No']['Mid_Salary']
Dev_onsite_salaries.to_csv('dev_onsite_salary.csv',index=False)
Dev_remote_salaries.to_csv('dev_remote_salary.csv',index=False)


from scipy.stats import median_test

# Perform Median Test
if not Dev_remote_salaries.empty and not Dev_onsite_salaries.empty:
    stat_dev, p_value_dev, _, _ = median_test(Dev_remote_salaries, Dev_onsite_salaries)
    print(f"Median Test statistic: {stat_dev}")
    print(f"P-value (Median Test): {p_value_dev}")

#---------------------------------------------------------------------------------

#Software Clound
cl_remote_salaries=Software_clound[Software_clound['Remote']=='Yes']['Mid_Salary']
cl_onsite_salaries=Software_Dev[Software_Dev['Remote'] == 'No']['Mid_Salary']

cl_remote_salaries.to_csv('cloud_remote_salary.csv',index=False)
cl_onsite_salaries.to_csv('cloud_onsite_salary.csv',index=False)

Software_clound[Software_clound['Remote']=='Yes']['Mid_Salary'].median()
Software_Dev[Software_Dev['Remote'] == 'No']['Mid_Salary'].median()

from scipy.stats import median_test

# Perform Median Test
if not cl_remote_salaries.empty and not cl_onsite_salaries.empty:
    stat, p_value, _, _ = median_test(cl_remote_salaries, cl_onsite_salaries)
    print(f"Median Test statistic: {stat}")
    print(f"P-value (Median Test): {p_value}")
    
#----------------------------------------------------------------

company_score_remote=data[data['Remote']=='Yes']['Company Score']
company_score_onsite=data[data['Remote']=='No']['Company Score']

company_score_remote.to_csv('company_score_remote.csv',index=False)
company_score_onsite.to_csv('company_score_onsite.csv',index=False)
from scipy import stats

# ทดสอบสมมติฐานทางเดียว (Remote > Onsite)
if not company_score_remote.empty and not company_score_onsite.empty:
    mann_greater_stat,mann_greater_p_values = stats.mannwhitneyu(company_score_remote, company_score_onsite, alternative='greater')
    print(f"Mann-Whitney U Statistic: {mann_greater_stat}")
    print(f"P-value (One-Tailed Test): {mann_greater_p_values}")

#------------------------------------------------------------------------------
from scipy import stats

# ทดสอบสมมติฐานทางเดียว (Remote > Onsite)
if not company_score_remote.empty and not company_score_onsite.empty:
    mann_left_stat,mann_left_p_values = stats.mannwhitneyu(company_score_remote, company_score_onsite, alternative='greater')
    print(f"Mann-Whitney U Statistic: {mann_left_stat}")
    print(f"P-value (One-Tailed Test): {mann_left_p_values}")
    


#------------------------------------------------------------------
data.info()
data['City'].fillna('Unknow', inplace=True)
data[data['City']=='Unknow']['City']
data['state'].fillna('Unknow',inplace=True)
data[data['state']=='Unknow']['state']

#---------------------------------------------------------------
import numpy as np
from scipy.stats import chisquare
# สร้างข้อมูลนับจำนวนการเกิดขึ้นจริงของแต่ละ state (จากคอลัมน์ 'state')
data['state'].value_counts()

observed_counts = data['state'].value_counts().values
observed_counts
# คาดว่าการกระจายตัวควรจะเท่าๆ กัน แต่ต้องแน่ใจว่าผลรวมเท่ากัน
expected_counts = np.full_like(observed_counts, np.mean(observed_counts))
# ปรับ expected_counts ให้มีผลรวมเท่ากับ observed_counts
expected_counts = expected_counts * (observed_counts.sum() / expected_counts.sum())
# เรียกใช้ Chi-Square Test
chi_stat, chi_p_value = chisquare(observed_counts, f_exp=expected_counts)
print(f"Chi-Square Statistic: {chi_stat}")
print(f"P-Value: {chi_p_value}")

#----------------------------------------------------------------------------
import numpy as np
from scipy.stats import chisquare

# สร้าง observed counts จากข้อมูล state (หรือข้อมูลหมวดหมู่อื่นๆ)
observed_counts = data['City'].value_counts().values
# คำนวณ expected counts แบบ uniform distribution
expected_counts = np.ones_like(observed_counts) * observed_counts.sum() / len(observed_counts)
# รวมเซลล์ที่มีค่า expected < 5
min_expected_threshold = 5
if np.any(expected_counts < min_expected_threshold):
    observed_counts = np.where(expected_counts < min_expected_threshold, 
                               observed_counts[expected_counts >= min_expected_threshold].sum(), 
                               observed_counts)
    expected_counts = np.where(expected_counts < min_expected_threshold, 
                               expected_counts[expected_counts >= min_expected_threshold].sum(), 
                               expected_counts)
# เรียกใช้ Chi-Square Test
chi_stat, chi_p_value = chisquare(observed_counts, f_exp=expected_counts)
print(f"Chi-Square Statistic: {chi_stat}")
print(f"P-Value: {chi_p_value}")

#------------------------------------------------------------------

import numpy as np
from scipy.stats import kstest, norm

# สมมติฐานว่าเงินเดือนในแบบ Remote และ Onsite ถูกดึงมา
s_remote_salaries = data[data['Remote'] == 'Yes']['Mid_Salary']
s_onsite_salaries = data[data['Remote'] == 'No']['Mid_Salary']

# ทดสอบการกระจายตัวของ Remote Salaries กับการกระจายปกติ
stat_remote, p_value_remote = kstest(s_remote_salaries, 'norm', args=(np.mean(s_remote_salaries),
                                                                      np.std(s_remote_salaries)))

# ทดสอบการกระจายตัวของ Onsite Salaries กับการกระจายปกติ
stat_onsite, p_value_onsite = kstest(s_onsite_salaries, 'norm', args=(np.mean(s_onsite_salaries), 
                                                                      np.std(s_onsite_salaries)))

# แสดงผลลัพธ์
print(f"Kolmogorov-Smirnov Test for Remote Salaries: Statistic={stat_remote}, P-Value={p_value_remote}")
print(f"Kolmogorov-Smirnov Test for Onsite Salaries: Statistic={stat_onsite}, P-Value={p_value_onsite}")

#-------------------------------------------------------

import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.runs import runstest_1samp

# สร้างตัวแปรบูลีนตามเงื่อนไขว่าเงินเดือนเกิน 100 หรือไม่
data['Above_100'] = data['Mid_Salary'] > 100

# ทำการทดสอบ Run Test
z_stat, p_value = runstest_1samp(data['Above_100'], correction=False)

# แสดงผลลัพธ์
print(f"Z-Statistic runs test: {z_stat}")
print(f"P-Value runs test: {p_value}")

#--------------------------------------------------

# นับจำนวนข้อมูลในแต่ละเมือง
city_counts = data['City'].value_counts()
city_counts
# เตรียมข้อมูลสำหรับการทดสอบ
city_counts = city_counts.sort_index()  # เรียงตามชื่อเมือง
city_counts_values = city_counts.values  # ค่า
n = len(city_counts_values)  # จำนวนเมือง
# เรียกใช้ Run Test for Randomness
z_stat_city, p_value_city = runstest_1samp(city_counts_values, correction=False)
# แสดงผลลัพธ์
print(f"Z-Statistic: {z_stat_city}")
print(f"P-Value: {p_value_city}")

#---------------------------------------------
# นับจำนวนข้อมูลในแต่ละรัฐ
state_counts = data['state'].value_counts()
state_counts
# เตรียมข้อมูลสำหรับการทดสอบ
state_counts = state_counts.sort_index()  # เรียงตามชื่อรัฐ
state_counts_values = state_counts.values  # ค่า
n = len(state_counts_values)  # จำนวนรัฐ
# เรียกใช้ Run Test for Randomness
z_stat_state, p_value_state = runstest_1samp(state_counts_values, correction=False)
# แสดงผลลัพธ์
print(f"Z-Statistic: {z_stat_state}")
print(f"P-Value: {p_value_state}")

#-----------------------------------------------
data.columns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
# สร้างโมเดล ANOVA
model = ols('Mid_Salary ~ C(City) + C(Company_Score_Group)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
# แสดงผล ANOVA table
print(anova_table)

import pandas as pd
from scipy.stats import friedmanchisquare
# จัดกลุ่มข้อมูลตาม Company_Score_Group และ City
data_grouped = data.groupby(['City', 'Company_Score_Group'])['Mid_Salary'].apply(list).reset_index()
data_grouped
# Pivot ข้อมูลเพื่อสร้าง DataFrame ที่เหมาะกับการทดสอบ Friedman Test
data_pivot = data_grouped.pivot(index='City', columns='Company_Score_Group', values='Mid_Salary')
data_pivot
# เติมข้อมูลที่ขาดหายด้วยค่าที่เหมาะสม (ถ้ามี)
data_pivot = data_pivot.applymap(lambda x: x if isinstance(x, list) else [0])
data_pivot
# ทดสอบ Friedman Test
grouped_data = [data_pivot[col].dropna().apply(lambda x: x[0]) for col in data_pivot.columns]
df=pd.DataFrame(grouped_data)
df.to_csv('2_way.csv',index=False)

# ตรวจสอบว่ามีกลุ่มข้อมูลครบถ้วน
if all(len(group) > 0 for group in grouped_data):
    statistic, p_value = friedmanchisquare(*grouped_data)
    print(f"Friedman Test Statistic: {statistic}")
    print(f"P-Value: {p_value}")


print(data)
data.columns
data.to_csv('data.csv',index=False)