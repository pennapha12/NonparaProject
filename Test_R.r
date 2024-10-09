#-----------------------------------------------------------------------------------------
# อ่านข้อมูลจากไฟล์ CSV ที่บันทึกไว้
en_salary <- read.csv('en_salary.csv')
dev_salary <- read.csv('dev_salary.csv')
cloud_salary <- read.csv('cloud_salary.csv')
other_salary <- read.csv('other_salary.csv')

# แสดงข้อมูลที่อ่านเข้ามา
print(en_salary)
print(dev_salary)
print(cloud_salary)
print(other_salary)


# รวมข้อมูลเข้าด้วยกันใน DataFrame เดียว
salary_data <- data.frame(
  Salary = c(en_salary$Mid_Salary, dev_salary$Mid_Salary, cloud_salary$Mid_Salary, other_salary$Mid_Salary),
  Job_Group = factor(rep(c("Software_En", "Software_Dev", "Software_Cloud", "Other"), 
                         c(nrow(en_salary), nrow(dev_salary), nrow(cloud_salary), nrow(other_salary))))
)
 salary_data

# ทำการทดสอบ Kruskal-Wallis
kruskal_test <- kruskal.test(Salary ~ Job_Group, data = salary_data)
# แสดงผลลัพธ์
print(kruskal_test)

#------------------------------------------------------------------------------------------------------

mid_salary <- read.csv('mid_salary.csv')
print(mid_salary)
company_score <- read.csv('company_score.csv')
print(company_score)

for_test_corr <- data.frame(
  mid_salary = c(mid_salary$Mid_Salary),
  company_score = c(company_score$Company.Score)
)

for_test_corr

# ทำการทดสอบ Pearson correlation ระหว่าง mid_salary และ company_score
pearson_test <- cor.test(for_test_corr$mid_salary, for_test_corr$company_score, method = "pearson")
# แสดงผลลัพธ์
print(pearson_test)

# ทำการทดสอบ Pearson correlation ระหว่าง mid_salary และ company_score
spearman_test <- cor.test(for_test_corr$mid_salary, for_test_corr$company_score, method = "spearman")
# แสดงผลลัพธ์
print(spearman_test)

#-------------------------------------------------------------------------------------------------------

# อ่านไฟล์ CSV และแสดงข้อมูล
contingency_table_sa_co <- read.csv('salary_score_range.csv')
print(contingency_table_sa_co)

# ทำการทดสอบ Chi-Square
chi_test <- chisq.test(contingency_table_sa_co)
# แสดงผลลัพธ์ของการทดสอบ
print(chi_test)
# แสดงผลลัพธ์ Expected Frequencies
cat("Expected frequencies:\n")
print(chi_test$expected)
# แสดงค่า Degrees of Freedom
cat("Degrees of freedom:", chi_test$parameter, "\n")
# แสดงค่า Chi-Square statistic
cat("Chi-Square statistic:", chi_test$statistic, "\n")
# แสดงค่า P-value
cat("P-value:", chi_test$p.value, "\n")

#------------------------------------------------------------------------------

install.packages("RVAideMemoire")
library(RVAideMemoire)
en_onsite_salary <- read.csv('en_onsite_salary.csv')
en_remote_salary <- read.csv('en_remote_salary.csv')
# รวมข้อมูลสองกลุ่มใน DataFrame เดียวกันและสร้างคอลัมน์กลุ่ม (Group)
for_en_med <- data.frame(
  salary = c(en_onsite_salary$Mid_Salary, en_remote_salary$Mid_Salary),
  group = factor(rep(c('Onsite', 'Remote'), 
                     c(nrow(en_onsite_salary), nrow(en_remote_salary))))
)
# ทดสอบค่ามัธยฐาน (Median Test)
res_en_med <- mood.medtest(salary ~ group, data = for_en_med)
# แสดงผลลัพธ์ของการทดสอบ
print(res_en_med)

#-----------------------------------------------------------

dev_onsite_salary <- read.csv('dev_onsite_salary.csv')
dev_remote_salary <- read.csv('dev_remote_salary.csv')
# รวมข้อมูลสองกลุ่มใน DataFrame เดียวกันและสร้างคอลัมน์กลุ่ม (Group)
for_dev_med <- data.frame(
  dev_salary = c(dev_onsite_salary$Mid_Salary, dev_remote_salary$Mid_Salary),
  dev_group = factor(rep(c('Onsite', 'Remote'), 
                     c(nrow(dev_onsite_salary), nrow(dev_remote_salary))))
)
# ทดสอบค่ามัธยฐาน (Median Test)
res_dev_med <- mood.medtest(dev_salary ~ dev_group, data = for_dev_med)
# แสดงผลลัพธ์ของการทดสอบ
print(res_dev_med)

#-------------------------------------------------------------------

cloud_onsite_salary <- read.csv('cloud_onsite_salary.csv')
cloud_remote_salary <- read.csv('dev_remote_salary.csv')
# รวมข้อมูลสองกลุ่มใน DataFrame เดียวกันและสร้างคอลัมน์กลุ่ม (Group)
for_cloud_med <- data.frame(
  cloud_salary = c(cloud_onsite_salary$Mid_Salary, cloud_remote_salary$Mid_Salary),
  cloud_group = factor(rep(c('Onsite', 'Remote'), 
                     c(nrow(cloud_onsite_salary), nrow(cloud_remote_salary))))
)
# ทดสอบค่ามัธยฐาน (Median Test)
res_cloud_med <- mood.medtest(cloud_salary ~ cloud_group, data = for_cloud_med)
# แสดงผลลัพธ์ของการทดสอบ
print(res_cloud_med)

#---------------------------------------------------------------------

# ทดสอบ Mann-Whitney U test
res_en_mw <- wilcox.test(en_onsite_salary$Mid_Salary, en_remote_salary$Mid_Salary, 
                         alternative = "two.sided")
# แสดงผลลัพธ์ของการทดสอบ
print(res_en_mw)

#-----------------------------------------------------------
 
# ทดสอบ Mann-Whitney U test
res_dev_mw <- wilcox.test(dev_onsite_salary$Mid_Salary, dev_remote_salary$Mid_Salary, 
                         alternative = "two.sided")
# แสดงผลลัพธ์ของการทดสอบ
print(res_dev_mw)

#----------------------------------------------------

# ทดสอบ Mann-Whitney U test
res_cloud_mw <- wilcox.test(cloud_onsite_salary$Mid_Salary, cloud_remote_salary$Mid_Salary, 
                         alternative = "two.sided")
# แสดงผลลัพธ์ของการทดสอบ
print(res_cloud_mw)

#-----------------------------------------------

company_score_onsite <- read.csv('company_score_onsite.csv')
company_score_remote <- read.csv('company_score_remote.csv')

# รวมข้อมูลและสร้าง DataFrame ใหม่
for_company_score_mwu <- data.frame(
  score = c(company_score_onsite$Company.Score, company_score_remote$Company.Score),
  group = factor(rep(c('Onsite', 'Remote'), 
                     c(nrow(company_score_onsite), nrow(company_score_remote))))
)


# ทดสอบ Mann-Whitney U Test
res_company_score_mwu <- wilcox.test(score ~ group, data = for_company_score_mwu , alternative='greater')
print(res_company_score_mwu)

#-------------------------------
# โหลดข้อมูล
data <- read.csv("data.csv")
# ทำ ANOVA โดยการใช้ City และ Company_Score_Group
anova_model <- aov(Mid_Salary ~ factor(City) + factor(Company_Score_Group), data = data)
anova_summary <- summary(anova_model)
# แสดงผล ANOVA table
print(anova_summary)

#--------------------------------------------------------
# โหลดข้อมูลที่จัดเรียงเป็นกลุ่มแล้ว
data_grouped <- read.csv("2_way.csv")
# ใช้ friedman.test ในการทดสอบ Friedman Test
friedman_test_result <- friedman.test(as.matrix(data_grouped))
# แสดงผล Friedman Test
print(friedman_test_result)

#----------------------------------------------------------------------------

# ติดตั้งแพ็คเกจที่จำเป็น (ถ้ายังไม่ได้ติดตั้ง)
# install.packages("dplyr")

library(dplyr)


# สร้างข้อมูลนับจำนวนการเกิดขึ้นจริงของแต่ละ state (จากคอลัมน์ 'state')
observed_counts <- table(data$state)
observed_counts
# คำนวณ expected counts โดยให้การกระจายตัวเท่าๆ กัน
expected_counts <- rep(mean(observed_counts), length(observed_counts))
expected_counts
# ปรับ expected_counts ให้มีผลรวมเท่ากับ observed_counts
expected_counts <- expected_counts * (sum(observed_counts) / sum(expected_counts))
expected_counts

# เรียกใช้ Chi-Square Test
chi_test <- chisq.test(observed_counts, p = expected_counts / sum(expected_counts))
# แสดงผลลัพธ์
cat("Chi-Square Statistic:", chi_test$statistic, "\n")
cat("P-Value:", chi_test$p.value, "\n")

#-------------------------------------

data <- read.csv("data.csv")
# สมมติฐานว่าเงินเดือนในแบบ Remote และ Onsite มีการแจกแจงแบบปกติหรือไม่
s_remote_salaries <- data[data$Remote == 'Yes', 'Mid_Salary']
s_onsite_salaries <- data[data$Remote == 'No', 'Mid_Salary']
# ทดสอบการกระจายตัวของ Remote Salaries กับการกระจายปกติ
ks_test_remote <- ks.test(s_remote_salaries, "pnorm", mean = mean(s_remote_salaries), sd = sd(s_remote_salaries))
# ทดสอบการกระจายตัวของ Onsite Salaries กับการกระจายปกติ
ks_test_onsite <- ks.test(s_onsite_salaries, "pnorm", mean = mean(s_onsite_salaries), sd = sd(s_onsite_salaries))
# แสดงผลลัพธ์
print(paste("Kolmogorov-Smirnov Test for Remote Salaries: Statistic =", ks_test_remote$statistic, 
", P-Value =", ks_test_remote$p.value))
print(paste("Kolmogorov-Smirnov Test for Onsite Salaries: Statistic =", ks_test_onsite$statistic, 
", P-Value =", ks_test_onsite$p.value))

#------------------------


# โหลดแพ็กเกจ tseries
library(tseries)
# สร้างตัวแปรบูลีนตามเงื่อนไขว่าเงินเดือนเกิน 100 หรือไม่
data$Above_100 <- data$Mid_Salary > 100
# ทำการทดสอบ Run Test
run_test_result <- runs.test(as.factor(data$Above_100))
# แสดงผลลัพธ์
print(run_test_result)

#-----------------------------------

# นับจำนวนข้อมูลในแต่ละเมือง
city_counts <- table(data$City)
# หาชื่อเมืองที่มีจำนวนมากที่สุด
most_frequent_city <- names(which.max(city_counts))
# เตรียมข้อมูลใหม่: เมืองที่มากที่สุด และอื่นๆ
new_city_counts <- c(sum(city_counts[names(city_counts) != most_frequent_city]), city_counts[most_frequent_city])
names(new_city_counts) <- c("Others", most_frequent_city)
# แสดงจำนวนข้อมูลในรูปแบบใหม่
print(new_city_counts)
# แปลงข้อมูลให้เป็น factor สำหรับ runs.test
new_city_counts_factor <- factor(rep(names(new_city_counts), new_city_counts))


# เรียกใช้ Run Test for Randomness
runs_test_result <- runs.test(new_city_counts_factor)
# แสดงผลลัพธ์
if (!is.null(runs_test_result)) {
    z_statistic <- sprintf("%.4f", runs_test_result$statistic)
    p_value <- sprintf("%.4f", runs_test_result$p.value)
    
    print(paste("Z-Statistic:", z_statistic))
    print(paste("P-Value:", p_value))
} else {
    print("Error: The runs test did not return a valid result.")
}
