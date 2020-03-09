from struct import unpack
from zlib import decompress
import sys
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass

@dataclass
class Definition:
    title: str
    entry_str: str
    parsed_entry: BeautifulSoup 


def gen_xml_entries(f):
    f.seek(0x40)
    limit = 0x40 + unpack('i', f.read(4))[0]
    f.seek(0x60)
    while f.tell()<limit:
        sz, = unpack('i', f.read(4))
        buf = decompress(f.read(sz)[8:])
        for m in re.finditer(b'<d:entry[^\n]+', buf):
            entry = m.group().decode()
            title = re.search('d:title="(.*?)"', entry).group(1)
            soup = BeautifulSoup(entry)
            
            yield Definition(
                title=title,
                entry_str=soup.get_text(),
                parsed_entry=soup,
            )
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Must have exactly one argument (input)")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        for definition in gen_xml_entries(f):
            print(definition.title)
            print(definition.entry_str)