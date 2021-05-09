import mysql.connector
import datetime
import time
import warnings
warnings.filterwarnings("ignore")
level=""
dept=""
t = time.localtime()
current_time = time.strftime("%H", t)

minet = time.strftime("%M", t)
minet=int(minet)
current_time=int(current_time)
if (current_time>=8 and current_time<9):
	current_time="08:00:00"
elif(current_time>=10 and current_time<11):
	current_time="10:00:00"
elif(current_time>=12 and current_time<13):
	current_time="12:00:00"
elif(current_time>=14 and current_time<15):
	current_time="14:00:00"
else:
	current_time="08:00:00"
dt = datetime.datetime.today()

year=dt.year
month=dt.month
day=dt.day

day_date = datetime.date(int(year), int(month), int(day))

day_name=day_date.strftime("%A")
r=["",""]
def main(students):

    db_connection = mysql.connector.connect(host="localhost",user="root",passwd="775092382ah96med", database="classroom_database")
    db_cursor = db_connection.cursor()

    for a in students:
        subject = ""
        level = ""
        department=""
        stid = ""


        db_cursor.execute("select levelid , departmentid from student where idstudent=%s",(str(a),))
        for table in db_cursor:
            level=table[0]
            department=table[1]
        db_cursor.execute("select it.subjectid   from lecture_table it ,  subject s , day d    where  it.subjectid= s.idsubject and it.dayid= d.idday and s.levelid=%s and s.departmentid=%s and d.dayname=%s and it.start_time=%s and it.roomid=%s ",(str(level),str(department),str(day_name),str(current_time),"100"))
        for table in db_cursor:
            subject=table[0]

        if subject!="":

            db_cursor.execute("select  ar.studentid from attendance_record ar where ar.studentid=%s and ar.subjectid=%s and ar.day_date=%s ", (str(a), str(subject), str(day_date)))
            for table in db_cursor:
                stid=table[0]

            if stid=="":
                db_cursor.execute("INSERT INTO attendance_record (studentid, subjectid, day_date) VALUES(%s, %s, %s)",(str(a), str(subject), str(day_date)))
                db_connection.commit()
                print("succesfuly to add into database")
