from requests_html import HTMLSession

import argparse

parser = argparse.ArgumentParser()
parser.parse_args()
parser.add_argument("url")

session = HTMLSession()
page = session.get(parser.url)
page.html.render()