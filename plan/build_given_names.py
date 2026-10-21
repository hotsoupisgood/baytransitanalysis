#!/usr/bin/env python3
"""
build_given_names.py — writes plan/given_names.txt, the lexicon behind the
`given_name` tier of pipeline/redact.py (CALIBRATION.md 2026-10-17 #2).

Source. No offline name corpus was installed on this Mac (no nltk corpora, no
faker, no names-dataset; pip is off-limits for the redaction pass), so the
list is written from general knowledge: roughly the ~800 most common US given
names of the last 70 years (SSA top lists, both sexes, with usual short forms)
plus common South Asian, East Asian, Southeast Asian, Middle Eastern and
Hispanic given names that appear in Bay Area public comment. It is NOT a
census extract and has no counts; it is a conservative recall aid, not a
definition of what a name is.

Flagging. A name that is also an ordinary English word or a place/brand/title
("Grant", "Hope", "Bill", "Austin", "Jordan") is written as `Name\tambiguous`:
the redactor never replaces it on its own, only as the first half of a two-
token full name ("Grant Hoffman"). The flag is seeded automatically from the
lower-case entries of /usr/share/dict/words (web2) at build time and then
corrected by hand in the two sets below; the OUTPUT file is what the redactor
reads, so the build is reproducible without the dictionary.

Re-run:  .venv/bin/python plan/build_given_names.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "given_names.txt"

US = """
James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony Mark
Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey Ryan Jacob Gary
Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel Gregory Alexander Frank Patrick Raymond
Jack Dennis Jerry Tyler Aaron Jose Adam Nathan Henry Douglas Zachary Peter Kyle Noah Ethan Jeremy Walter
Christian Keith Roger Terry Austin Sean Gerald Carl Harold Dylan Arthur Lawrence Jordan Jesse Bryan Billy Bruce
Gabriel Joe Logan Alan Juan Albert Willie Elijah Wayne Randy Vincent Mason Roy Ralph Bobby Russell Bradley
Philip Eugene Louis Howard Harry Victor Jimmy Martin Johnny Craig Phillip Clarence Fred Ernest Leonard Stanley
Glenn Earl Dale Allen Marvin Norman Herbert Lee Leroy Francis Melvin Curtis Frederick Eddie Chris Clifford
Alfred Bernard Warren Ray Calvin Leon Lloyd Floyd Gordon Milton Edwin Dean Clyde Vernon Jim Tom Bill Bob Mike
Steve Dave Dan Ken Ted Ed Al Ben Sam Max Jon Rick Rich Greg Jeff Matt Nick Tony Andy Tim Pat Rob Doug Chuck
Jay Ron Don Phil Lou Hank Len Les Ned Ted Vic Wes Will Pete Art Hal Gus Abe Ike Ira Sid Sol Stu Ty Jed Bud
Liam Oliver Lucas Levi Owen Wyatt Caleb Isaac Luke Carter Jayden Hunter Connor Landon Evan Isaiah Cameron
Nolan Hudson Colton Easton Parker Cooper Ian Brayden Dominic Adrian Xavier Jaxon Leo Miles Eli Max Asher
Micah Jace Grayson Lincoln Jackson Sawyer Bentley Declan Everett Finn Theo Archer Emmett Rowan Silas Jasper
Ezra Axel Beau Brooks Kai Ryder Tucker Weston Zane Colin Blake Chase Cole Cody Corey Derek Devin Drew Dustin
Garrett Grant Jared Jesse Jordan Kurt Lance Marcus Mitchell Nathaniel Neil Oscar Preston Reed Shane Shawn
Spencer Taylor Travis Trevor Troy Vance Wade Wesley Brent Brett Byron Casey Clint Dwight Elliott Emmanuel Erik
Fernando Forrest Gavin Graham Grady Jamie Jonah Josiah Julian Kirk Lane Lyle Malcolm Marshall Maxwell Morgan
Nelson Orlando Perry Quinn Reginald Rex Rodney Roland Rudy Sidney Simon Sterling Stuart Toby Tristan Vaughn
Wendell Zachariah Abraham Alvin Angelo Barry Bart Brady Brock Bryce Cedric Clayton Clifton Dallas Damon Darrell
Darren Darryl Dewey Dexter Dirk Donovan Dudley Duncan Edgar Elmer Elton Emmanuel Ernie Felix Fletcher Gene Gil
Gilbert Guy Hector Horace Hugh Hugo Irving Ivan Jerome Judd Julius Kelvin Kent Kirby Lamar Lester Lionel Lonnie
Luther Lyndon Mack Marlon Maurice Monty Morris Murray Myron Nigel Otis Pablo Percy Quentin Ramon Randall Rufus
Rupert Sheldon Sherman Stewart Terrence Theodore Virgil Wallace Wilbur Wilson Winston Woodrow Zeke
Mary Patricia Jennifer Linda Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Margaret Sandra Ashley
Kimberly Emily Donna Michelle Carol Amanda Dorothy Melissa Deborah Stephanie Rebecca Sharon Laura Cynthia Kathleen
Amy Angela Shirley Anna Brenda Pamela Emma Nicole Helen Samantha Katherine Christine Debra Rachel Carolyn Janet
Catherine Maria Heather Diane Ruth Julie Olivia Joyce Virginia Victoria Kelly Lauren Christina Joan Evelyn Judith
Megan Andrea Cheryl Hannah Jacqueline Martha Gloria Teresa Ann Sara Madison Frances Kathryn Janice Jean Abigail
Alice Judy Sophia Grace Denise Amber Doris Marilyn Danielle Beverly Isabella Theresa Diana Natalie Brittany
Charlotte Marie Kayla Alexis Lori Jane Louise Julia Rose Phyllis Irene Lillian Ruby Gladys Lois Edna Edith
Ethel Thelma Florence Mildred Bernice Hazel Beatrice Bertha Agnes Ella Eleanor Audrey Esther Eva Geraldine
Hattie Ida Jeanette Lucille Marion Marjorie Mattie Minnie Myrtle Nellie Norma Pauline Pearl Rosa Sylvia Vera
Viola Vivian Wanda Wilma Ada Alma Bessie Blanche Clara Cora Daisy Della Dora Elsie Emma Erma Eunice Fannie
Flora Georgia Harriet Hilda Inez Iva Jessie Josephine Juanita Laverne Leona Lena Lillie Lola Lorraine Lula
Mabel Mae Maggie Mamie Maude Nora Olga Opal Ora Rosie Sadie Stella Velma Willie Winifred Kim Sue Jan Jo Pam
Peggy Deb Debbie Cathy Kathy Cindy Patty Patti Terri Sherry Vicki Vickie Tammy Tina Lynn Lynne Gail Gayle Jill
Jackie Becky Jenny Kristen Kristin Kristi Krista Jodi Jody Lori Traci Tracy Staci Stacy Stacey Wendy Robin
Tonya Tanya Tara Teri Toni Bonnie Connie Ronnie Lonnie Dawn Faith Hope Joy April May June Summer Autumn Holly
Ivy Lily Violet Jasmine Jade Crystal Amber Misty Candy Cherry Ginger Brandy Sandy Mandy Randi Kelli Kelley
Shelley Shelly Sheila Shauna Shawna Shannon Erin Megan Meghan Kerry Kerri Tracey Courtney Whitney Brooke Paige
Morgan Taylor Jordan Casey Jamie Jessie Dana Leslie Lesley Carey Kelsey Lindsay Lindsey Mackenzie Madison Haley
Hailey Hayley Kaitlyn Caitlin Katelyn Kaylee Kylie Riley Sydney Zoe Chloe Ava Mia Sofia Avery Ella Scarlett
Aria Layla Lily Nora Hazel Luna Ellie Aubrey Addison Natalie Leah Savannah Brooklyn Stella Claire Skylar Lucy
Paisley Anna Caroline Nova Genesis Emilia Kennedy Samantha Maya Willow Kinsley Naomi Aaliyah Elena Sarah
Ariana Allison Gabriella Alice Madelyn Cora Ruby Eva Serenity Autumn Adeline Hailey Gianna Valentina Isla
Eliana Quinn Nevaeh Ivy Sadie Piper Lydia Alexa Josephine Emery Julia Delilah Arianna Vivian Kaylee Sophie
Brielle Madeline Peyton Rylee Clara Hadley Melanie Mackenzie Reagan Adalynn Liliana Aurora Everly Ariel
Trinity Harmony Destiny Melody Liberty Charity Patience Mercy Prudence Justice Heaven Angel Diamond Precious
Bridget Colleen Maureen Eileen Kathleen Noreen Moira Siobhan Fiona Deirdre Megan Tara Shannon Kelly Erin
Agatha Antoinette Bernadette Camille Celeste Chantal Danielle Genevieve Giselle Jacqueline Josette Juliette
Madeleine Margot Monique Nadine Noelle Simone Suzanne Yvette Yvonne Adele Anita Annette Arlene Bonita Carla
Carmen Cecilia Celia Charlene Darlene Elaine Estelle Francine Georgina Gretchen Harriet Heidi Ingrid Irma
Jeanne Joanne Joanna Johanna Kristina Lorena Lucinda Marcia Marsha Marlene Maxine Miriam Muriel Naomi Nina
Paula Pauline Priscilla Ramona Regina Renee Rhonda Roberta Rochelle Rosalind Rosemary Roxanne Sabrina Selma
Sondra Sonia Sonya Tabitha Tamara Trudy Ursula Valerie Vanessa Veronica Wilhelmina Yolanda Zelda
Alexandra Alyssa Amelia Angelina Annie Ariel Ashlee Ashlyn Bailey Bethany Bianca Briana Brianna Brianne
Brittney Callie Camila Carly Cassandra Cassidy Cassie Chelsea Cheyenne Christy Claudia Dakota Daniela Desiree
Destiny Dominique Elise Eliza Erica Erika Felicia Gabrielle Gina Gwen Gwendolyn Haley Hillary Hilary Jada
Jenna Jillian Jocelyn Jolene Josie Juliana Kara Karina Karla Kasey Kate Katie Kathy Kayla Kaylee Keisha Kendra
Kiara Kirsten Kristy Kyla Lacey Lara Latoya Laurie Leanne Lena Lila Lilly Lorraine Lucia Lyla Macy Maddie
Mallory Marcy Margo Marissa Marisa Marla Marlena Maryann Meagan Melinda Meredith Mia Mindy Molly Monica
Nadia Natasha Nichole Nikki Olivia Peyton Phoebe Rae Raquel Reba Regina Rhiannon Rita Robyn Rosalie Rosanna
Roxanne Sally Selena Serena Shana Shari Sharon Shawna Shelby Sheri Sherri Sierra Stacie Suzette Sydney Tabatha
Tasha Tatiana Tess Tessa Tiffany Tori Trisha Trista Valarie Vicky Viviana Yasmin Yesenia Zoey
Abby Addie Alexis Alanna Allie Amie Anya Beth Betsy Billie Bobbie Brandi Bree Cara Carol Carrie Chrissy
Dee Dixie Dolly Dottie Effie Elsa Etta Evie Fern Frankie Gabby Gerri Gigi Greta Hattie Jeri Jess Joni Jules
Kat Kay Kit Kitty Lea Leigh Lexi Liz Lizzie Lulu Mandi Marnie Mavis Millie Mimi Nan Nell Nettie Patsy Penny
Polly Rhoda Roz Sal Sammy Sasha Tilly Trixie Vi Winnie Zelda
"""

SOUTH_ASIAN = """
Aarav Aditya Ajay Akash Amit Amar Anand Anil Ankit Ankur Anuj Arjun Arun Arvind Ashok Ashish Ashwin Atul Balaji
Bharat Chetan Deepak Dev Dhruv Dinesh Gaurav Girish Gopal Harish Hari Harsh Hemant Jatin Jay Kapil Karan Karthik
Kiran Krishna Kunal Lalit Mahesh Manish Manoj Mayank Mohan Mohit Mukesh Naveen Neeraj Nikhil Niraj Nitin Pankaj
Parag Pradeep Prakash Pramod Pranav Prasad Praveen Prashant Rahul Raj Rajesh Rajiv Rajeev Rakesh Ram Ramesh Ravi
Rishi Rohan Rohit Sachin Sameer Samir Sandeep Sanjay Sanjeev Santosh Satish Saurabh Shailesh Shankar Sharad
Shashank Shiv Shyam Siddharth Sridhar Srinivas Subhash Sudhir Sunil Suresh Tarun Tushar Uday Umesh Varun
Venkat Vijay Vikas Vikram Vinay Vinod Vipul Vishal Vivek Yash Yogesh Aamir Abdul Adnan Ahmed Ahmad Ali Amir
Arif Asad Asif Bilal Farhan Faisal Hamza Hassan Hussain Ibrahim Imran Irfan Jamal Kamran Khalid Mohammed
Muhammad Mohammad Mustafa Nadeem Naveed Omar Osman Rashid Saad Saeed Salman Shahid Tariq Usman Waseem Yusuf
Zafar Zahid Aisha Amina Anjali Ankita Anita Anu Aparna Archana Arti Aarti Asha Bhavna Chitra Deepa Deepika
Divya Durga Gayatri Geeta Gita Hema Indira Jaya Jyoti Kajal Kalpana Kavita Kavya Kiran Komal Kriti Lakshmi
Lata Leela Madhu Madhuri Mala Mamta Manisha Maya Meena Meera Megha Mira Mona Monika Namrata Nandini Neha
Neelam Neeta Nidhi Nisha Nita Padma Pallavi Parul Pooja Poonam Prachi Preeti Priya Priyanka Puja Radha Rachna
Rajni Rakhi Rani Ranjana Rashmi Rekha Renu Richa Rina Ritu Roshni Ruchi Rupa Sadhana Sangeeta Sapna Sarita
Seema Shalini Shanti Sheetal Shikha Shilpa Shobha Shreya Shruti Simran Sita Smita Sneha Sonal Sonali Sonia
Sudha Sujata Suman Sunita Sushma Swati Tanvi Tara Trisha Uma Usha Vandana Vani Varsha Veena Vidya Yamini
Fatima Farah Hina Huma Iqra Khadija Mariam Maryam Nadia Nasreen Rabia Saba Sadia Saima Sana Shazia Sobia
Sumaira Tahira Uzma Zainab Zara Zeenat Zoya
"""

EAST_ASIAN = """
Wei Jing Li Ming Hui Fang Yan Hong Jun Lei Xin Yu Yi Ying Jie Ping Qing Mei Ling Lin Hua Jian Bin Bo Chen
Cheng Feng Gang Guo Hai Hao Jia Jin Kai Kun Lan Liang Long Min Ning Peng Qiang Rui Shan Tao Ting Wen Xia
Xiang Xiao Xiu Xue Yang Yao Ye Yong Yuan Yun Zhen Zhi Zhong Zhou Zhang Xiaoming Xiaoli Xiaoyan Jiawei
Wenjie Yifan Yiming Zihan Ziyi Yuxin Yuting Haoran Hanyu Jiayi Jingyi Mengyao Qianyi Ruoxi Shuang Siyuan
Tianyi Xinyi Yutong Zeyu Kenji Hiroshi Takashi Takeshi Taro Jiro Ichiro Kazuo Akira Satoshi Masaru Yoshi
Yoshio Yuki Yuji Yusuke Daisuke Ryo Ryota Shinji Shota Sho Tatsuya Kenta Kenta Haruto Hinata Yuto Ren Sora
Kaito Riku Daiki Naoki Ryosuke Yukiko Akiko Keiko Yoko Yoshiko Kazuko Noriko Michiko Emiko Sachiko Mariko
Tomoko Junko Naoko Aiko Ayumi Haruka Yui Yuna Sakura Hana Misaki Mio Rin Aoi Mei Nana Hinata Koharu Rina
Minho Minjun Jihoon Jiho Seojun Hajun Junseo Sungmin Sunghoon Hyunwoo Dongwoo Jaehyun Jiwon Jisoo Jiyeon
Jiyoung Soyeon Seoyeon Seoyoon Hayoon Hyejin Eunji Eunjung Yejin Yoona Sujin Sumin Minji Minseo Jiwoo Yuna
Haeun Chaeyoung Sooyoung Nguyen Anh Minh Hung Huy Khoa Khang Linh Thu Thao Trang Phuong Lan Hoa Mai Ngoc
Quynh Tuan Thanh Trung Duc Dung Hieu Hoang Long Nam Phong Quang Son Thai Tien Tri Vinh Vu Bao Binh Chi Dat
Hanh Hien Hong Huong Kim Loan Ly Nga Nhung Oanh Thuy Trinh Van Vy Xuan Yen Jose Ramon Maricel Marites
Rosalinda Jocelyn Rowena Lourdes Dolores Nenita Angelita Rodel Joel Bong Boyet Jun Rey Arnel Noel Lito
Somchai Somsak Nattapong Supaporn Siriporn Malee Niran Pornthip Kanya Chai Anong Arun Budi Agus Dewi Siti
Putri Rina Ayu Wati Sri Indah Eko Wahyu Bambang Joko Adi Rizky Ahmad Dimas Fajar Rudi Hendra
"""

HISPANIC = """
Jose Juan Luis Carlos Jorge Miguel Pedro Francisco Antonio Manuel Rafael Ricardo Roberto Fernando Alejandro
Alberto Alfredo Andres Angel Armando Arturo Cesar Cristian Daniel David Diego Eduardo Emilio Enrique Ernesto
Esteban Felipe Gerardo Gilberto Gonzalo Guillermo Gustavo Hector Hugo Humberto Ignacio Ismael Jaime Javier
Jesus Joaquin Julio Leonardo Lorenzo Marco Marcos Mario Martin Mateo Mauricio Moises Nicolas Octavio Omar
Oscar Pablo Ramiro Ramon Raul Rene Rigoberto Rodolfo Rodrigo Rogelio Rolando Ruben Salvador Santiago Sergio
Tomas Ulises Vicente Victor Wilfredo Xavier Adrian Alonso Benito Bernardo Candido Celso Cruz Dario Efrain
Eliseo Eusebio Fabian Fidel Florencio Gregorio Heriberto Isidro Ivan Lazaro Leopoldo Lucio Marcelino Marcelo
Maximo Modesto Nestor Noe Pascual Porfirio Reynaldo Rosendo Santos Saul Teodoro Tiburcio Valentin Venancio
Ariel Elias Jonathan Josue Kevin Bryan Brayan Yahir Jair Emiliano Matias Thiago Sebastian Santiago Maria
Guadalupe Rosa Carmen Ana Juana Margarita Josefina Teresa Martha Luz Elena Patricia Gloria Veronica Leticia
Lourdes Dolores Silvia Alicia Adriana Angelica Beatriz Blanca Carolina Catalina Cecilia Claudia Consuelo
Cristina Daniela Diana Elisa Elvira Esperanza Estela Eva Fabiola Fernanda Gabriela Graciela Griselda Irma
Isabel Jacqueline Jessica Juanita Karina Laura Lidia Liliana Lorena Lucia Lupe Magdalena Marcela Maribel
Maricela Marisol Marta Mayra Mercedes Micaela Miriam Monica Natalia Nora Norma Olga Paola Paula Perla Pilar
Ramona Raquel Rebeca Reyna Rocio Rosalba Rosario Sandra Sara Socorro Sofia Soledad Susana Tania Valeria
Viviana Ximena Xochitl Yadira Yesenia Yolanda Yvette Zulema Alejandra Alma Amparo Araceli Aurora Azucena
Belen Berta Camila Celina Concepcion Delia Dulce Edith Elsa Emma Erika Esmeralda Estrella Evelyn Flor
Herminia Hilda Ines Itzel Jazmin Julieta Leonor Lizbeth Lucero Luisa Manuela Mariana Marina Matilde Nancy
Nayeli Nelly Noemi Ofelia Otilia Paulina Regina Renata Rosalia Rosaura Ruth Salma Sonia Trinidad Victoria
Virginia Yareli Yazmin Zoila Chuy Pepe Paco Nacho Lalo Memo Beto Toño Chucho Lupita Chayo Concha Maru
"""

# Hand corrections to the automatic (dictionary) flag.
# force_ambiguous: not a lower-case dictionary word but still a place, brand,
# title, abbreviation or another proper noun that is frequent in these meetings.
FORCE_AMBIGUOUS = set("""
Austin Madison Lincoln Jackson Jordan Dallas Orlando Kent Troy Savannah Brooklyn Chelsea Sydney Sierra Dakota
Georgia Virginia Victoria Florence Carolina Montana Nevada Phoenix Houston Kennedy Carter Cooper Hunter Mason
Parker Porter Archer Baker Fletcher Sawyer Tucker Taylor Morgan Marshall Wilson Nelson Sterling Preston Everett
Easton Weston Hudson Brooks Lane Park Bay Hill Grant Chase Hope Grace Will Bill Mark Reed Wade Dean Gene Frank
Earl Lance Miles Clay Cliff Glenn Glen Dale Ray Rich Rob Pat Art Max Guy Jay Sue Kit Kay Joy Faith Dawn Rose
Lily Violet Daisy Holly Ivy Olive Penny Robin Ruby Pearl Crystal Amber Jade Summer Autumn May June April
Candy Cherry Ginger Brandy Sherry Sandy Misty Angel Destiny Harmony Melody Liberty Charity Patience Mercy
Prudence Justice Heaven Diamond Precious Genesis Trinity Serenity Nova Luna Aurora Willow Hazel Piper Quinn
Paisley Scarlett Ariel Eden Journey Royal Prince King Duke Bishop Major Marshal Judge Chance Cash Colt Dash
Stone Page Sage Scout Skip Spike Red Woody Rocky Rusty Dusty Bud Buck Chip Jack Jimmy Jenny Billy Bobby Bob
Nick Mike Tom Don Josh Chuck Dick Peter John Matt Mat Jean Dot Fay Carol Victor Warren Norm Herb Pete Terry
Perry Harry Tony Randy Dolly Kitty Sally Polly Patty Patti Dixie Fern Nan Nell Sal Vi Jo Jan Ed Al Lou Len
Les Ned Vic Wes Sid Sol Stu Ty Jed Ike Ira Abe Hal Gus Dee Bo Lee Lin Long Min Ning Shan Ting Wen Bin Hai
Jin Kai Lan Tao Yang Ye Yu Yi Li Ming Chen Cheng Feng Gang Guo Hao Hong Jun Kun Peng Rui Xia Yao Yong Yuan Yun
Zhou Zhang Yoshi Ren Rin Mei Nana Hana Sora Kim Anh Linh Mai Van Nam Son Thai Phong Tien Ly Loan Chi Dat Vu
Bao Binh Hanh Hong Huong Nga Trinh Vy Xuan Yen Nguyen Cruz Santos Jesus Angel Ramon Mateo Pablo Rafael Carlos
Lorenzo Antonio Martin Jose Clara Rosa Francisco Bruno Leandro Anselmo Teresa Luz Pilar Rosario Trinidad
Socorro Concepcion Flor Dulce Estrella Reyna Rey Noel Jun Bong Joel Chai Arun Dev Raj Ram Hari Jay Kiran Uma
Asha Maya Mira Mona Tara Rani Puja Sita Dinesh Prasad Shiv Krishna Ali Omar Ahmed Amir Jamal Hassan Hamza
Abdul Mohammed Sana Nadia Zara Rose Belen Alma Nora Emma Eva Ruth Edith Elsa Sonia Nancy Laura Sara Diana
Elena Ana Maria Jessica Victoria Virginia Regina Camila Erika Sofia Juanita Lupe Jazmin Monica Paula Olga
Norma Irma Marina Marta
""".split())
# force_name: dictionary words that are overwhelmingly a person in meeting
# speech and that the hand read showed are worth redacting bare.
FORCE_NAME = set("""
Jennifer Patricia Barbara Elizabeth Susan Margaret Dorothy Deborah Stephanie Rebecca Kathleen Amy Angela
Shirley Anna Brenda Pamela Michelle Nicole Samantha Katherine Christine Debra Rachel Carolyn Janet Catherine
Heather Diane Julie Olivia Joyce Kelly Lauren Christina Joan Evelyn Judith Megan Andrea Cheryl Hannah Martha
Gloria Ann Kathryn Janice Abigail Alice Judy Sophia Denise Doris Marilyn Danielle Beverly Theresa Natalie
Brittany Charlotte Marie Kayla Alexis Lori Jane Louise Julia Phyllis Irene Lillian Gladys Lois Edna Ethel
Thelma Mildred Bernice Beatrice Bertha Agnes Ella Eleanor Audrey Esther Geraldine Ida Lucille Marion Marjorie
Mattie Minnie Myrtle Nellie Pauline Sylvia Vera Viola Vivian Wanda Wilma Ada Bessie Blanche Cora Della Dora
Elsie Eunice Fannie Flora Harriet Hilda Inez Josephine Leona Lena Lillie Lola Lorraine Mabel Mae Maggie
Mamie Maude Opal Ora Rosie Sadie Stella Velma Winifred Peggy Debbie Cathy Kathy Cindy Terri Vicki Tammy Tina
Lynn Gail Jill Jackie Becky Kristen Kristin Jodi Jody Tracy Stacy Stacey Wendy Tonya Tanya Teri Toni Bonnie
Connie Ronnie Lonnie Sheila Shannon Erin Kerry Courtney Whitney Brooke Paige Dana Leslie Kelsey Lindsay
Lindsey Haley Kaitlyn Caitlin Kylie Riley Zoe Chloe Ava Mia Avery Aria Layla Ellie Aubrey Addison Leah Claire
Lucy Caroline Emilia Naomi Aaliyah Ariana Allison Gabriella Madelyn Adeline Gianna Valentina Isla Eliana
Sadie Lydia Alexa Emery Delilah Arianna Sophie Brielle Madeline Peyton Rylee Hadley Melanie Mackenzie Reagan
Liliana Everly Bridget Colleen Maureen Eileen Moira Fiona Deirdre Camille Celeste Genevieve Giselle Juliette
Margot Monique Nadine Noelle Simone Suzanne Yvonne Adele Anita Annette Arlene Carla Carmen Cecilia Celia
Charlene Darlene Elaine Estelle Francine Gretchen Heidi Ingrid Jeanne Joanne Joanna Johanna Kristina Lorena
Lucinda Marcia Marsha Marlene Maxine Miriam Muriel Nina Priscilla Ramona Renee Rhonda Roberta Rochelle
Rosalind Rosemary Roxanne Sabrina Selma Sonya Tabitha Tamara Trudy Ursula Valerie Vanessa Veronica Yolanda
Alexandra Alyssa Amelia Angelina Annie Bethany Bianca Briana Brianna Callie Carly Cassandra Cassidy Cassie
Cheyenne Claudia Daniela Desiree Dominique Elise Eliza Erica Felicia Gabrielle Gina Gwen Gwendolyn Jada Jenna
Jillian Jocelyn Jolene Josie Juliana Kara Karina Karla Kate Katie Keisha Kendra Kiara Kirsten Lacey Lara
Laurie Leanne Lila Lilly Lucia Lyla Macy Mallory Marcy Marissa Marisa Marla Meredith Melinda Mindy Molly
Natasha Nikki Phoebe Raquel Rita Robyn Rosalie Selena Serena Shari Shawna Shelby Sheri Sherri Tasha Tatiana
Tess Tessa Tiffany Tori Trisha Vicky Viviana Yasmin Yesenia Zoey Abby Addie Allie Beth Betsy Billie Bobbie
Bree Cara Carrie Chrissy Dottie Effie Etta Evie Frankie Gabby Gerri Gigi Greta Hattie Jeri Jess Joni Jules
Kat Lea Leigh Lexi Liz Lizzie Lulu Marnie Mavis Millie Mimi Nettie Patsy Rhoda Roz Sammy Sasha Tilly Trixie
Winnie Zelda
James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony
Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey Ryan Jacob
Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel Gregory Alexander Patrick
Raymond Dennis Jerry Tyler Aaron Adam Nathan Henry Douglas Zachary Kyle Noah Ethan Jeremy Walter Christian
Keith Roger Sean Gerald Carl Harold Dylan Arthur Lawrence Jesse Bryan Bruce Gabriel Joe Logan Alan Juan Albert
Willie Elijah Wayne Vincent Roy Ralph Russell Bradley Philip Eugene Louis Howard Phillip Clarence Fred Ernest
Leonard Stanley Allen Marvin Norman Herbert Leroy Francis Melvin Curtis Frederick Eddie Chris Clifford Alfred
Bernard Calvin Leon Lloyd Floyd Gordon Milton Edwin Clyde Vernon Jim Steve Dave Dan Ken Ted Ben Sam Jon Rick
Greg Jeff Andy Tim Doug Ron Phil Hank Liam Oliver Lucas Levi Owen Wyatt Caleb Isaac Luke Jayden Connor Landon
Evan Isaiah Cameron Nolan Colton Ian Brayden Dominic Adrian Xavier Jaxon Leo Eli Asher Micah Jace Grayson
Bentley Declan Finn Theo Emmett Rowan Silas Jasper Ezra Axel Beau Ryder Zane Colin Blake Cole Cody Corey Derek
Devin Drew Dustin Garrett Jared Kurt Marcus Mitchell Nathaniel Neil Oscar Shane Shawn Spencer Travis Trevor
Vance Wesley Brent Brett Byron Casey Clint Dwight Elliott Emmanuel Erik Fernando Forrest Gavin Graham Grady
Jamie Jonah Josiah Julian Kirk Lyle Malcolm Maxwell Rodney Roland Rudy Sidney Simon Stuart Toby Tristan Vaughn
Wendell Zachariah Abraham Alvin Angelo Barry Bart Brady Brock Bryce Cedric Clayton Clifton Damon Darrell
Darren Darryl Dewey Dexter Dirk Donovan Dudley Duncan Edgar Elmer Elton Ernie Felix Gil Gilbert Hector Horace
Hugh Hugo Irving Ivan Jerome Judd Julius Kelvin Kirby Lamar Lester Lionel Luther Lyndon Mack Marlon Maurice
Monty Morris Murray Myron Nigel Otis Percy Quentin Randall Rufus Rupert Sheldon Sherman Stewart Terrence
Theodore Virgil Wallace Wilbur Winston Woodrow Zeke
""".split())

SHORT_MIN = 3   # two-letter names (Al, Ed, Jo, Bo, Vi, Li, Yu, Yi, Ye) are never redacted bare


def main():
    words = set()
    try:
        for w in Path("/usr/share/dict/words").read_text(errors="replace").split():
            if w[:1].islower():
                words.add(w)
    except OSError:
        pass
    seen, rows = set(), []
    for block, tag in ((US, "us"), (SOUTH_ASIAN, "south-asian"), (EAST_ASIAN, "east-asian"), (HISPANIC, "hispanic")):
        for n in block.split():
            key = n.lower()
            if key in seen:
                continue
            seen.add(key)
            amb = (key in words and n not in FORCE_NAME) or n in FORCE_AMBIGUOUS or len(n) < SHORT_MIN
            rows.append((n, tag, amb))
    rows.sort(key=lambda r: r[0].lower())
    n_amb = sum(1 for r in rows if r[2])
    head = [
        "# plan/given_names.txt — lexicon for the `given_name` tier of pipeline/redact.py.",
        "# Generated by plan/build_given_names.py (read its docstring for the source: written",
        "# from general knowledge, NOT a census/SSA extract; no offline corpus was installed).",
        "# One name per line, tab-separated: Name<TAB>origin-tag<TAB>flag. flag `ambiguous` =",
        "# also an English word, place, brand, title or 2-letter form: never redacted on its own,",
        "# only as the first token of a two-token full name. flag `name` = redacted bare.",
        f"# {len(rows)} names, {n_amb} ambiguous.",
    ]
    OUT.write_text("\n".join(head) + "\n" + "".join(f"{n}\t{t}\t{'ambiguous' if a else 'name'}\n" for n, t, a in rows),
                   encoding="utf-8")
    print(f"wrote {OUT}: {len(rows)} names, {n_amb} ambiguous")


if __name__ == "__main__":
    main()
