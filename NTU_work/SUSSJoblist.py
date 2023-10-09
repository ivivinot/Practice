# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:36:30 2023

@author: iru-ra2
"""
#this code work
import requests 
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date

today = date.today()
today = today.strftime('%Y%m%d')

#URL for the job list
url_1 = "https://careers.suss.edu.sg/go/View-All-Jobs/4640510/"
url_2 = "https://careers.suss.edu.sg/go/View-All-Jobs/4640510/25"
url_3 = "https://careers.suss.edu.sg/go/View-All-Jobs/4640510/50"

def collect_jobtitles(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup=BeautifulSoup(response.text, "html.parser")
        job_title_elements = soup.find_all(class_="jobTitle hidden-phone")
        job_facilty_elements = soup.find_all('td', class_="colFacility hidden-phone")
        job_org_elements = soup.find_all('td', class_="colDepartment hidden-phone")
        job_URL_elements = soup.find_all('span', class_="jobTitle visible-phone")



        job_titles = [element.get_text(strip=True) for element in job_title_elements]
        job_facilities = [element.get_text(strip=True) for element in job_facilty_elements]
        job_orgs = [element.get_text(strip=True) for element in job_org_elements]
        job_URL = []
        for element in job_URL_elements:
            job_links = element.find_all('a')
            for job_link in job_links:
                job_URL.append("https://careers.suss.edu.sg"+ job_link.get('href'))
        return job_titles, job_facilities, job_orgs, job_URL
    else:
        print("Failed to retrieve {url}")
        return[]

def fill_missing_data(data_list, target_length):
    if len(data_list) < target_length:
        data_list.extend(['N/A']) * (target_length - len(data_list))

#collect job titles from both page
jobtitle_page1,jobfacilities_page1, joborgs_page1, jobURL_page1 = collect_jobtitles(url_1)
jobtitle_page2,jobfacilities_page2, joborgs_page2, jobURL_page2 = collect_jobtitles(url_2)
jobtitle_page3,jobfacilities_page3, joborgs_page3, jobURL_page3 = collect_jobtitles(url_3)

#combine the job titles from both page into one list
all_jobURL         = jobURL_page1 + jobURL_page2 + jobURL_page3
all_jobtitles      = jobtitle_page1 + jobtitle_page2 + jobtitle_page3
all_jobfacilities  = jobfacilities_page1 + jobfacilities_page2 + jobfacilities_page3
all_joborgs        = joborgs_page1 + joborgs_page2 + joborgs_page3

max_length = len(all_jobtitles)
fill_missing_data(all_jobURL, max_length)
fill_missing_data(all_jobfacilities, max_length)
fill_missing_data(all_joborgs, max_length)

#check for success in data pulling
#for index,joburl in enumerate(all_joborgs, start = 1):
#    print(f"{index}.{joburl}")
   

#Create a Dataframe to store the job titles
df = pd.DataFrame({'Job Title': all_jobtitles,
                   'Field' : all_jobfacilities,
                   'Organisation' : all_joborgs,
                   'Job URL' : all_jobURL})

print("transfer of job to dataframe")
#Save the Dataframe to an Excel files
excel_file = "SUSS joblisting" + today + ".xlsx"
df.to_excel(excel_file, index=False)


