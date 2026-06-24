import gradio as gr
print('Gradio version:', gr.__version__)
import app
print('Calling launch...')
app.demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True)
