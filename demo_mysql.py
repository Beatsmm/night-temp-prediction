import pymysql

def print_data():
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='Nov2014', database='mugu_blog_article', charset='utf8')
    cur = conn.cursor()
    cur.execute('select * from article')

    data = cur.fetchall()
    for d in data:
        print(d)

    cur.close()
    conn.close()



if __name__ == '__main__':
    print_data()