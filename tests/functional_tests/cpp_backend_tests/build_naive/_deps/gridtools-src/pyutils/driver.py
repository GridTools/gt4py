#!/usr/bin/env python3

import json
import os

from pyutils import args, env, log

script_dir = os.path.dirname(os.path.abspath(__file__))


@args.command(description='main script for GridTools pyutils')
@args.arg('--verbose',
          '-v',
          action='count',
          default=0,
          help='increase verbosity (use -vv for all debug messages)')
@args.arg('--logfile', '-l', help='path to logfile')
def driver(verbose, logfile):
    log.set_verbosity(verbose)
    if logfile:
        log.log_to_file(logfile)


@driver.command(description='build GridTools')
@args.arg('--build-type', '-b', choices=['release', 'debug'], required=True)
@args.arg('--environment', '-e', help='path to environment file')
@args.arg('--target', '-t', nargs='+', help='make targets to build')
@args.arg('--source-dir', help='GridTools source directory')
@args.arg('--build-dir', '-o', required=True, help='build directory')
@args.arg('--install-dir', '-i', help='install directory')
@args.arg('--cmake-only',
          action='store_true',
          help='only execute CMake but do not build')
def build(build_type, environment, target, source_dir, build_dir, install_dir,
          cmake_only):
    import build

    if source_dir is None:
        source_dir = os.path.abspath(os.path.join(script_dir, os.path.pardir))

    env.set_cmake_arg('CMAKE_BUILD_TYPE', build_type.title())

    if environment:
        env.load(environment)

    build.cmake(source_dir, build_dir, install_dir)
    if not cmake_only:
        build.make(build_dir, target)


try:
    from pyutils import buildinfo
except ImportError:
    buildinfo = None

if buildinfo:

    @driver.command(description='run GridTools tests')
    @args.arg('--run-mpi-tests',
              '-m',
              action='store_true',
              help='enable execution of MPI tests')
    @args.arg('--perftests-only',
              action='store_true',
              help='only run perftests binaries')
    @args.arg('--verbose-ctest',
              action='store_true',
              help='run ctest in verbose mode')
    @args.arg('--examples-build-dir',
              help='build directory for examples',
              default=os.path.join(buildinfo.binary_dir, 'examples_build'))
    @args.arg('--build-examples',
              '-b',
              action='store_true',
              help='enable building of GridTools examples')
    def test(run_mpi_tests, perftests_only, verbose_ctest, examples_build_dir,
             build_examples):
        import test

        if perftests_only:
            test.run_perftests()
        else:
            test.run(run_mpi_tests, verbose_ctest)

        if build_examples:
            test.compile_and_run_examples(examples_build_dir, verbose_ctest)


@driver.command(description='performance test utilities')
def perftest():
    pass


if buildinfo:

    @perftest.command(description='run performance tests')
    @args.arg('--domain-size',
              '-s',
              required=True,
              type=int,
              nargs=3,
              metavar=('ISIZE', 'JSIZE', 'KSIZE'),
              help='domain size (excluding halo)')
    @args.arg('--runs',
              default=100,
              type=int,
              help='number of runs to do for each stencil')
    @args.arg('--output',
              '-o',
              required=True,
              help='output file path, extension .json is added if not given')
    def run(domain_size, runs, output):

        import perftest
        if not output.lower().endswith('.json'):
            output += '.json'

        data = perftest.run(domain_size, runs)
        with open(output, 'w') as outfile:
            json.dump(data, outfile, indent='  ')
            log.info(f'Successfully saved perftests output to {output}')


@perftest.command(description='plot performance results')
def plot():
    pass


def _load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


@plot.command(description='plot performance comparison')
@args.arg('--output', '-o', required=True, help='output directory')
@args.arg('--input', '-i', required=True, nargs=2, help='two input files')
def compare(output, input):
    from perftest import plot

    plot.compare(*(_load_json(i) for i in input), output)


@plot.command(description='plot performance history')
@args.arg('--output', '-o', required=True, help='output directory')
@args.arg('--input',
          '-i',
          required=True,
          nargs='+',
          help='any number of input files')
@args.arg('--date',
          '-d',
          default='job',
          choices=['commit', 'job'],
          help='date to use, either the build/commit date or the date when '
          'the job was run')
@args.arg('--limit',
          '-l',
          type=int,
          help='limit the history size to the given number of results')
def history(output, input, date, limit):
    from perftest import plot

    plot.history([_load_json(i) for i in input], output, date, limit)


@plot.command(description='plot backends comparison')
@args.arg('--output', '-o', required=True, help='output directory')
@args.arg('--input',
          '-i',
          required=True,
          nargs='+',
          help='any number of input files')
def compare_backends(output, input):
    from perftest import plot

    plot.compare_backends([_load_json(i) for i in input], output)


with log.exception_logging():
    driver()
