import argparse
import sys
sys.path.append('energy_sampling')
sys.argv = ['energy_sampling/train.py']
exec(open('energy_sampling/train.py').read().split('if __name__')[0])
print("epochs default:", parser._defaults.get('epochs') or 
      [a for a in parser._actions if '--epochs' in str(a.option_strings)][0].default)
