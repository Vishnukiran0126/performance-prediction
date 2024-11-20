from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Initialize WebDriver (e.g., using ChromeDriver)
driver = webdriver.Chrome()  # You can also use Firefox or any other supported driver

# Open the target page
driver.get("https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;type=batting")

# Wait for the page to load (adjust the time if necessary)
time.sleep(3)  # Wait 3 seconds for the page to load

# Find the player name input field by XPath
player_name_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div[1]/div[3]/div[3]/form/input[1]")

# Enter the player's name (replace with the name given by the user)
player_name_input.send_keys('Virat Kohli')  # You can replace 'Virat Kohli' with the user's input

# Find the submit button by its class name or XPath
submit_button = driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div[1]/div[3]/div[3]/form/input[2]")

# Click the submit button to search for the player
submit_button.click()

# Wait for the results to load (adjust the time if necessary)
time.sleep(5)  # Adjust the time depending on the page load time

odilink=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/div[4]/table/tbody/tr/td[3]/a[2]")
# Optionally, you can check the result or print out the page title after submission
odilink.click()
#time.sleep(4)

radiobtn=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/table[1]/tbody/tr/td/table/tbody/tr[2]/td/form/table/tbody/tr[10]/td[2]/table/tbody/tr[3]/td[1]/label[3]/input")
radiobtn.click()

subquery=driver.find_element(By.XPATH,"/html/body/div[3]/div[1]/div[1]/div[3]/div/table[1]/tbody/tr/td/table/tbody/tr[2]/td/form/table/tbody/tr[11]/td[2]/table/tbody/tr/td[1]/input")
subquery.click()
print(driver.title)  # This will print the title of the results page

# Close the browser once done
driver.quit()
