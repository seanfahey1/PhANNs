import html
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory , Markup
import subprocess

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__)) 
UPLOAD_FOLDER = ROOT_FOLDER + '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'faa', 'fasta', 'gif', 'fa'])
import urllib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['APPLICATION_ROOT']='/adrian_net'
PREFIX=app.config['APPLICATION_ROOT'] 

def fix_url_for(path, **kwargs):
    return PREFIX + url_for(path, **kwargs)

#make fix_url_for available in tamplates
@app.context_processor
def contex():
    return dict(fix_url_for = fix_url_for)

#add the sorable attribute to tables generated by pandas
@app.template_filter('sorttable')
def sorttable_filter(s):
	s= s.replace('table id=','table class="sortable" id=')
	return s


def return_html_table(filename):
    cmd = ["python" , "run_di_model.py" , app.config['UPLOAD_FOLDER'] + '/' + filename]
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    out,err = p.communicate()
    print(err)
    return out


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/test')
def test_template():
    return "mira un salmon"

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(fix_url_for('uploaded_file',
                                    filename=filename))
    print( fix_url_for('upload_file'))
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html', title='about')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('index.html', table_code= Markup(return_html_table(filename).decode('utf8')))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
    #app.run(debug=True, host="0.0.0.0", port=8080)
    app.run(host="0.0.0.0", port=8080)
