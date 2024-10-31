# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score

from sklearn.base import clone
from collections import defaultdict
import warnings
import signal
import ctypes
import threading
from logging import NullHandler
import functools
import logging
LOG = logging.getLogger(name='stopit')
tid_ctype = ctypes.c_ulong
LOG.addHandler(NullHandler())
class TimeoutException(Exception):
    """Raised when the block under context management takes longer to complete
    than the allowed maximum timeout value.
    """
    pass

class BaseTimeout(object):
    """Context manager for limiting in the time the execution of a block
    :param seconds: ``float`` or ``int`` duration enabled to run the context
      manager block
    :param swallow_exc: ``False`` if you want to manage the
      ``TimeoutException`` (or any other) in an outer ``try ... except``
      structure. ``True`` (default) if you just want to check the execution of
      the block with the ``state`` attribute of the context manager.
    """
    # Possible values for the ``state`` attribute, self explanative
    EXECUTED, EXECUTING, TIMED_OUT, INTERRUPTED, CANCELED = range(5)

    def __init__(self, seconds, swallow_exc=True):
        self.seconds = seconds
        self.swallow_exc = swallow_exc
        self.state = BaseTimeout.EXECUTED

    def __bool__(self):
        return self.state in (BaseTimeout.EXECUTED, BaseTimeout.EXECUTING, BaseTimeout.CANCELED)

    __nonzero__ = __bool__  # Python 2.x

    def __repr__(self):
        """Debug helper
        """
        return "<{0} in state: {1}>".format(self.__class__.__name__, self.state)

    def __enter__(self):
        self.state = BaseTimeout.EXECUTING
        self.setup_interrupt()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is TimeoutException:
            if self.state != BaseTimeout.TIMED_OUT:
                self.state = BaseTimeout.INTERRUPTED
                self.suppress_interrupt()
            LOG.warning("Code block execution exceeded {0} seconds timeout".format(self.seconds),
                        exc_info=(exc_type, exc_val, exc_tb))
            return self.swallow_exc
        else:
            if exc_type is None:
                self.state = BaseTimeout.EXECUTED
            self.suppress_interrupt()
        return False

    def cancel(self):
        """In case in the block you realize you don't need anymore
       limitation"""
        self.state = BaseTimeout.CANCELED
        self.suppress_interrupt()

    # Methods must be provided by subclasses
    def suppress_interrupt(self):
        """Removes/neutralizes the feature that interrupts the executed block
        """
        raise NotImplementedError

    def setup_interrupt(self):
        """Installs/initializes the feature that interrupts the executed block
        """
        raise NotImplementedError


class base_timeoutable(object):  # noqa
    """A base for function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.
    Any decorated callable may receive a ``timeout`` optional parameter that
    specifies the number of seconds allocated to the callable execution.
    The decorated functions that exceed that timeout return ``None`` or the
    value provided by the decorator.
    :param default: The default value in case we timed out during the decorated
      function execution. Default is None.
    :param timeout_param: As adding dynamically a ``timeout`` named parameter
      to the decorated callable may conflict with the callable signature, you
      may choose another name to provide that parameter. Your decoration line
      could look like ``@timeoutable(timeout_param='my_timeout')``
    .. note::
       This is a base class that must be subclassed. subclasses must override
       thz ``to_ctx_mgr`` with a timeout  context manager class which in turn
       must subclasses of above ``BaseTimeout`` class.
    """
    to_ctx_mgr = None

    def __init__(self, default=None, timeout_param='timeout'):
        self.default, self.timeout_param = default, timeout_param

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timeout = kwargs.pop(self.timeout_param, None)
            if timeout:
                with self.to_ctx_mgr(timeout, swallow_exc=True):
                    result = self.default  # noqa
                    # ``result`` may not be assigned below in case of timeout
                    result = func(*args, **kwargs)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper

def async_raise(target_tid, exception):
    """Raises an asynchronous exception in another thread.
    Read http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
    for further enlightenments.
    :param target_tid: target thread identifier
    :param exception: Exception class to be raised in that thread
    """
    # Ensuring and releasing GIL are useless since we're not in C
    # gil_state = ctypes.pythonapi.PyGILState_Ensure()
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_ctype(target_tid),
                                                     ctypes.py_object(exception))
    # ctypes.pythonapi.PyGILState_Release(gil_state)
    if ret == 0:
        raise ValueError("Invalid thread ID {}".format(target_tid))
    elif ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_ctype(target_tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadingTimeout(BaseTimeout):
    """Context manager for limiting in the time the execution of a block
    using asynchronous threads launching exception.
    See :class:`stopit.utils.BaseTimeout` for more information
    """
    def __init__(self, seconds, swallow_exc=True):
        super(ThreadingTimeout, self).__init__(seconds, swallow_exc)
        self.target_tid = threading.current_thread().ident
        self.timer = None  # PEP8

    def stop(self):
        """Called by timer thread at timeout. Raises a Timeout exception in the
        caller thread
        """
        self.state = BaseTimeout.TIMED_OUT
        async_raise(self.target_tid, TimeoutException)

    # Required overrides
    def setup_interrupt(self):
        """Setting up the resource that interrupts the block
        """
        self.timer = threading.Timer(self.seconds, self.stop)
        self.timer.start()

    def suppress_interrupt(self):
        """Removing the resource that interrupts the block
        """
        self.timer.cancel()


class threading_timeoutable(base_timeoutable):  #noqa
    """A function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.
    this one uses ``ThreadingTimeout`` context manager.
    See :class:`.utils.base_timoutable`` class for further comments.
    """
    to_ctx_mgr = ThreadingTimeout


class SignalTimeout(BaseTimeout):
    """Context manager for limiting in the time the execution of a block
    using signal.SIGALRM Unix signal.
    See :class:`stopit.utils.BaseTimeout` for more information
    """
    def __init__(self, seconds, swallow_exc=True):
        seconds = int(seconds)  # alarm delay for signal MUST be int
        super(SignalTimeout, self).__init__(seconds, swallow_exc)

    def handle_timeout(self, signum, frame):
        self.state = BaseTimeout.TIMED_OUT
        raise TimeoutException('Block exceeded maximum timeout '
                               'value (%d seconds).' % self.seconds)

    # Required overrides
    def setup_interrupt(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def suppress_interrupt(self):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


class signal_timeoutable(base_timeoutable):  #noqa
    """A function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.
    this one uses ``SignalTimeout`` context manager.
    See :class:`.utils.base_timoutable`` class for further comments.
    """
    to_ctx_mgr = SignalTimeout


def pick_two_individuals_eligible_for_crossover(population):
    """Pick two individuals from the population which can do crossover, that is, they share a primitive.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    tuple: (individual, individual)
        Two individuals which are not the same, but share at least one primitive.
        Alternatively, if no such pair exists in the population, (None, None) is returned instead.
    """
    primitives_by_ind = [set([node.name for node in ind if isinstance(node, gp.Primitive)])
                         for ind in population]
    pop_as_str = [str(ind) for ind in population]

    eligible_pairs = [(i, i+1+j) for i, ind1_prims in enumerate(primitives_by_ind)
                                 for j, ind2_prims in enumerate(primitives_by_ind[i+1:])
                                 if not ind1_prims.isdisjoint(ind2_prims) and
                                    pop_as_str[i] != pop_as_str[i+1+j]]

    # Pairs are eligible in both orders, this ensures that both orders are considered
    eligible_pairs += [(j, i) for (i, j) in eligible_pairs]

    if not eligible_pairs:
        # If there are no eligible pairs, the caller should decide what to do
        return None, None

    pair = np.random.randint(0, len(eligible_pairs))
    idx1, idx2 = eligible_pairs[pair]

    return population[idx1], population[idx2]


def mutate_random_individual(population, toolbox):
    """Picks a random individual from the population, and performs mutation on a copy of it.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    individual: individual
        An individual which is a mutated copy of one of the individuals in population,
        the returned individual does not have fitness.values
    """
    idx = np.random.randint(0,len(population))
    ind = population[idx]
    ind, = toolbox.mutate(ind)
    del ind.fitness.values
    return ind


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    offspring = []

    for _ in range(lambda_):
        op_choice = np.random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = pick_two_individuals_eligible_for_crossover(population)
            if ind1 is not None:
                ind1_cx, _, evaluated_individuals_= toolbox.mate(ind1, ind2)
                del ind1_cx.fitness.values

                if str(ind1_cx) in evaluated_individuals_:
                    ind1_cx = mutate_random_individual(population, toolbox)
                offspring.append(ind1_cx)
            else:
                # If there is no pair eligible for crossover, we still want to
                # create diversity in the population, and do so by mutation instead.
                ind_mu = mutate_random_individual(population, toolbox)
                offspring.append(ind_mu)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = mutate_random_individual(population, toolbox)
            offspring.append(ind)
        else:  # Apply reproduction
            idx = np.random.randint(0, len(population))
            offspring.append(toolbox.clone(population[idx]))

    return offspring

def initialize_stats_dict(individual):
    '''
    Initializes the stats dict for individual
    The statistics initialized are:
        'generation': generation in which the individual was evaluated. Initialized as: 0
        'mutation_count': number of mutation operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'crossover_count': number of crossover operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'predecessor': string representation of the individual. Initialized as: ('ROOT',)

    Parameters
    ----------
    individual: deap individual

    Returns
    -------
    object
    '''
    individual.statistics['generation'] = 0
    individual.statistics['mutation_count'] = 0
    individual.statistics['crossover_count'] = 0
    individual.statistics['predecessor'] = 'ROOT',


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar,
                   stats=None, halloffame=None, verbose=0,
                   per_generation_function=None, log_file=None):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param pbar: processing bar
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param per_generation_function: if supplied, call this function before each generation
                            used by tpot to save best pipeline before each new generation
    :param log_file: io.TextIOWrapper or io.StringIO, optional (defaul: sys.stdout)
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initialize statistics dict for the individuals in the population, to keep track of mutation/crossover operations and predecessor relations
    for ind in population:
        initialize_stats_dict(ind)

    population[:] = toolbox.evaluate(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)


        # Update generation statistic for all individuals which have invalid 'generation' stats
        # This hold for individuals that have been altered in the varOr function
        for ind in offspring:
            if ind.statistics['generation'] == 'INVALID':
                ind.statistics['generation'] = gen

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        offspring = toolbox.evaluate(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # pbar process
        if not pbar.disable:
            # Print only the best individual fitness
            if verbose == 2:
                high_score = max(halloffame.keys[x].wvalues[1] \
                    for x in range(len(halloffame.keys)))
                pbar.write('\nGeneration {0} - Current '
                            'best internal CV score: {1}'.format(gen,
                                                        high_score),

                            file=log_file)

            # Print the entire Pareto front
            elif verbose == 3:
                pbar.write('\nGeneration {} - '
                            'Current Pareto front scores:'.format(gen),
                            file=log_file)
                for pipeline, pipeline_scores in zip(halloffame.items, reversed(halloffame.keys)):
                    pbar.write('\n{}\t{}\t{}'.format(
                            int(pipeline_scores.wvalues[0]),
                            pipeline_scores.wvalues[1],
                            pipeline
                        ),
                        file=log_file
                    )

        # after each population save a periodic pipeline
        if per_generation_function is not None:
            per_generation_function(gen)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    return population, logbook


def cxOnePoint(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    common_types = []
    for idx, node in enumerate(ind2[1:], 1):
        if node.ret in types1 and node.ret not in types2:
            common_types.append(node.ret)
        types2[node.ret].append(idx)

    if len(common_types) > 0:
        type_ = np.random.choice(common_types)

        index1 = np.random.choice(types1[type_])
        index2 = np.random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


# point mutation function
def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive no matter if it has the same number of arguments from the :attr:`pset`
    attribute of the individual.
    Parameters
    ----------
    individual: DEAP individual
        A list of pipeline operators and model parameters that can be
        compiled by DEAP into a callable function

    Returns
    -------
    individual: DEAP individual
        Returns the individual with one of point mutation applied to it

    """

    index = np.random.randint(0, len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)

    if node.arity == 0:  # Terminal
        term = np.random.choice(pset.terminals[node.ret])
        if isclass(term):
            term = term()
        individual[index] = term
    else:   # Primitive
        # find next primitive if any
        rindex = None
        if index + 1 < len(individual):
            for i, tmpnode in enumerate(individual[index + 1:], index + 1):
                if isinstance(tmpnode, gp.Primitive) and tmpnode.ret in node.args:
                    rindex = i
                    break

        # pset.primitives[node.ret] can get a list of the type of node
        # for example: if op.root is True then the node.ret is Output_DF object
        # based on the function _setup_pset. Then primitives is the list of classifor or regressor
        primitives = pset.primitives[node.ret]

        if len(primitives) != 0:
            new_node = np.random.choice(primitives)
            new_subtree = [None] * len(new_node.args)
            if rindex:
                rnode = individual[rindex]
                rslice = individual.searchSubtree(rindex)
                # find position for passing return values to next operator
                position = np.random.choice([i for i, a in enumerate(new_node.args) if a == rnode.ret])
            else:
                position = None
            for i, arg_type in enumerate(new_node.args):
                if i != position:
                    term = np.random.choice(pset.terminals[arg_type])
                    if isclass(term):
                        term = term()
                    new_subtree[i] = term
            # paste the subtree to new node
            if rindex:
                new_subtree[position:position + 1] = individual[rslice]
            # combine with primitives
            new_subtree.insert(0, new_node)
            individual[slice_] = new_subtree

    return individual,


@threading_timeoutable(default="Timeout")
def _wrapped_cross_val_score(sklearn_pipeline, features, target,
                             cv, scoring_function, sample_weight=None,
                             groups=None, use_dask=False):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    sklearn_pipeline : pipeline object implementing 'fit'
        The object to use to fit the data.
    features : array-like of shape at least 2D
        The data to fit.
    target : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    cv: cross-validation generator
        Object to be used as a cross-validation generator.
    scoring_function : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    sample_weight : array-like, optional
        List of sample weights to balance (or un-balanace) the dataset target as needed
    groups: array-like {n_samples, }, optional
        Group labels for the samples used while splitting the dataset into train/test set
    use_dask : bool, default False
        Whether to use dask
    """
    sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)

    features, target, groups = indexable(features, target, groups)

    cv_iter = list(cv.split(features, target, groups))
    scorer = check_scoring(sklearn_pipeline, scoring=scoring_function)

    if use_dask:
        try:
            import dask_ml.model_selection  # noqa
            import dask  # noqa
            from dask.delayed import Delayed
        except Exception as e:
            msg = "'use_dask' requires the optional dask and dask-ml depedencies.\n{}".format(e)
            raise ImportError(msg)

        dsk, keys, n_splits = dask_ml.model_selection._search.build_graph(
            estimator=sklearn_pipeline,
            cv=cv,
            scorer=scorer,
            candidate_params=[{}],
            X=features,
            y=target,
            groups=groups,
            fit_params=sample_weight_dict,
            refit=False,
            error_score=float('-inf'),
        )

        cv_results = Delayed(keys[0], dsk)
        scores = [cv_results['split{}_test_score'.format(i)]
                  for i in range(n_splits)]
        CV_score = dask.delayed(np.array)(scores)[:, 0]
        return dask.delayed(np.nanmean)(CV_score)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = [_fit_and_score(estimator=clone(sklearn_pipeline),
                                         X=features,
                                         y=target,
                                         scorer=scorer,
                                         train=train,
                                         test=test,
                                         verbose=0,
                                         parameters=None,
                                         error_score='raise',
                                         fit_params=sample_weight_dict)
                                    for train, test in cv_iter]
                if isinstance(scores[0], list): #scikit-learn <= 0.23.2
                    CV_score = np.array(scores)[:, 0]
                elif isinstance(scores[0], dict): # scikit-learn >= 0.24
                    from sklearn.model_selection._validation import _aggregate_score_dicts
                    CV_score = _aggregate_score_dicts(scores)["test_scores"]
                else:
                    raise ValueError("Incorrect output format from _fit_and_score!")
                CV_score_mean = np.nanmean(CV_score)
            return CV_score_mean
        except TimeoutException:
            return "Timeout"
        except Exception as e:
            return -float('inf')
