from flask import Flask, render_template, request, redirect, url_for, session, send_file
from script import gen_save_img

app = Flask(__name__)
called_gen = True
num_tri = 0
num_rect = 0

def gen():
    global num_tri, num_rect
    num_tri, num_rect = gen_save_img(1, 'static')
    print(f'result tri(gen):{num_tri}')
    print(f'result rect(gen):{num_rect}')
    return num_tri, num_rect

gen()

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == "POST":
        username = 'test'
        password = 'test'
        global called_gen, num_tri, num_rect

        userget = request.form['username']
        passget = request.form['password']
        print(f'user:{userget}')
        print(f'pass:{passget}')

        captcha_resulttri = int(request.form['triangle'])
        captcha_resultrect = int(request.form['rectangle'])
        print(f'answer tri:{captcha_resulttri}')
        print(f'answer rect:{captcha_resultrect}')

        if userget == username and passget == password:
            if captcha_resulttri == num_tri and captcha_resultrect == num_rect:
            # Authentication successful
                print('pass')
                return redirect(url_for('dashboard'))
            else:
            # Authentication failed
                print('fail')
                called_gen = False
                if called_gen == False:
                    gen()
                    called_gen = True
                return render_template('login.html', error='Captcha verification failed.')
        else:
            return render_template('login.html', error='Incorrect username or password.')
    else:
        return render_template('login.html')

@app.route('/captcha_image')
def captcha_image():
    img_path = 'static/processing_1.png'
    return send_file(img_path, mimetype='image/png')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/refresh')
def refresh():
    print('refresh')
    global called_gen
    called_gen = False
    if called_gen == False:
        gen()
        called_gen = True
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
