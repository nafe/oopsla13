#!/usr/bin/env python

import getopt
import subprocess
import sys
import os

AXIOMS_DIR = 'axioms'
KERNEL     = 'kernel.cl'

class BINOP(object):
  ADD = 'BINOP_ADD'
  MAX = 'BINOP_MAX'
  OR  = 'BINOP_OR'
  ABS = 'BINOP_ABSTRACT'

class CHECK(object):
  RACE      = 'CHECK_RACE'
  BI        = 'CHECK_BI'
  BI_ACCESS = 'CHECK_BI_ACCESS'

class PART(object):
  UPSWEEP   = 'INC_UPSWEEP'
  DOWNSWEEP = 'INC_DOWNSWEEP'
  ENDSPEC   = 'INC_ENDSPEC'

class SPEC(object):
  THREAD  = 'SPEC_THREADWISE'
  ELEMENT = 'SPEC_ELEMENTWISE'

class Options(object):
  N = 4
  op = BINOP.ADD
  width = 32
  verbose = False
  parts = []
  spec = SPEC.THREAD
  memout = 8000 # megabytes
  timeout = 3600 # seconds
  relentless = False
  repeat = 0
  specs_dir = 'specs'

def ispow2(x):
  return x != 0 and ((x & (x-1)) == 0)

def help(progname):
  print 'SYNOPSIS: Run staged verification on algorithm'
  print
  print 'USAGE: %s [options] n' % progname
  print
  print '  -h             Display this message'
  print '  --verbose      Show commands to run'
  print '  --op=X         Choose binary operator'
  print '  --width=X      Choose bitwidth'
  print '  --memout=X'
  print '  --timeout=X'
  return 0

def error(msg):
  print 'ERROR: %s' % msg
  return 1

def main(argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]
  try:
    opts, args = getopt.getopt(argv[1:],'h',
      ['verbose','help',
       'op=','width=',
       'memout=','timeout=','relentless','repeat=',
      ])
  except getopt.GetoptError:
    return error('error parsing options; try -h')
  for o, a in opts:
    if o in ('-h','--help'):
      return help(progname)
    if o == "--verbose":
      Options.verbose = True
    if o == "--op":
      op = a.lower()
      if op == 'add':        Options.op = BINOP.ADD
      elif op == 'max':      Options.op = BINOP.MAX
      elif op == 'or':       Options.op = BINOP.OR
      elif op == 'abstract': Options.op = BINOP.ABS
      else: return error('operator [%s] not recognised' % a)
    if o == "--width":
      try:
        width = int(a)
      except ValueError:
        return error('width must be an integer')
      if width not in [8,16,32]:
        return error('width must be one of 8, 16, 32')
      Options.width = width
    if o == '--memout':
      try:
        Options.memout = int(a)
      except ValueError:
        return error('bad memout [%s] given' % a)
    if o == '--timeout':
      try:
        Options.timeout = int(a)
      except ValueError:
        return error('bad timeout [%s] given' % a)
    if o == "--relentless":
      Options.relentless = True
    if o == "--repeat":
      try:
        Options.repeat = int(a)
      except ValueError:
        return error('bad repeat [%s] given' % a)
  if len(args) != 1:
    return error('number of elements not specified')
  try:
    Options.N = int(args[0])
  except ValueError:
    return error('invalid value given for n')
  if not ispow2(Options.N):
    return error('n must be a power of two')
  if not (2 <= Options.N and Options.N <= 128):
    return error('n must be between 2 and 128')
  Options.parts = [PART.UPSWEEP, PART.DOWNSWEEP]
  build_and_run([CHECK.RACE, CHECK.BI_ACCESS])
  checks = [CHECK.BI]
  extraflags = ['--no-barrier-access-checks','--only-divergence','--asymmetric-asserts']
  Options.parts = [PART.UPSWEEP]
  build_and_run(checks,extraflags)
  Options.parts = [PART.DOWNSWEEP]
  build_and_run(checks,extraflags)
  if Options.op != BINOP.ABS:
    Options.parts = [PART.ENDSPEC]
    build_and_run(checks,extraflags)
  return 0

def buildcmd(checks,extraflags=[]):
  cmd = [ 'gpuverify',
          '--silent',
          '--time-as-csv=%s' % fname(),
          '--testsuite',
          '--no-infer',
          '--no-source-loc-infer',
          '--only-intra-group',
          '--timeout=%d' % Options.timeout,
          '-I%s' % Options.specs_dir,
          '-DN=%d' % (Options.N*2),
          '-D%s' % Options.op,
          '-Ddwidth=32',
          '-Drwidth=%d' % Options.width,
        ]
  if Options.memout > 0:
    cmd.append('--memout=%d' % Options.memout)
  if PART.ENDSPEC in Options.parts:
    cmd.append('-D%s' % Options.spec)
  cmd.extend(['-D%s' % x for x in Options.parts])
  cmd.extend(['-D%s' % x for x in checks])
  if Options.op == BINOP.ABS and CHECK.BI in checks:
    if PART.UPSWEEP in Options.parts: bpl = 'upsweep'
    elif PART.DOWNSWEEP in Options.parts: bpl = 'downsweep'
    cmd.append('--boogie-file=%s/%s%d.bpl' % (AXIOMS_DIR,bpl,Options.width))
  cmd.extend(extraflags)
  cmd.append(KERNEL)
  return cmd

def fname(suffix=None):
  def aux(x): return x.split('_')[1].lower()
  op = aux(Options.op)
  part = '_'.join([aux(x) for x in Options.parts])
  if part == 'upsweep_downsweep': part = 'race-biacc'
  if Options.width == 32: ty = 'uint'
  elif Options.width == 16: ty = 'ushort'
  elif Options.width == 8: ty = 'uchar'
  else: assert False
  if suffix: return '%04d-%s-%s-%s.%s' % (Options.N,op,part,ty,suffix)
  else: return '%04d-%s-%s-%s' % (Options.N,op,part,ty)

def run(cmd):
  if Options.verbose: print ' '.join(cmd)
  code = subprocess.call(cmd)
  return code

def build_and_run(checks,extraflags=[]):
  cmd = buildcmd(checks,extraflags)
  code = run(cmd)
  if code != 0:
    print 'Problem during verification... aborting'
    sys.exit(1)

if __name__ == '__main__':
  sys.exit(main())
