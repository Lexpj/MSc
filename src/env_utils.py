def identify_env(env_id):

    CLASSIC_CONTROL = [
      'acrobot',
      'cartpole',
      'mountaincar',
      'mountaincarcontinuous',
      'pendulum'
    ]
    MUJOCO = [
      'ant',
      'halfcheetah',
      'hopper',
      'humanoidstandup',
      'humnaoid',
      'inverteddoublependulum',
      'invertedpendulum',
      'pusher',
      'reacher',
      'swimmer',
      'walker2d'
    ]
    ATARI = [
      'adventure',
      'airraid',
      'alien',
      'amidar',
      'assault',
      'asterix',
      'asteroids',
      'atlantis',
      'atlantis2',
      'backgammon',
      'bankheist',
      'basicmath',
      'battlezone',
      'beamrider',
      'berzerk',
      'blackjack',
      'bowling',
      'boxing',
      'breakout',
      'carnival',
      'casino',
      'centipede',
      'choppercommand',
      'crazyclimber',
      'crossbow',
      'darkchambers',
      'defender',
      'demonattack',
      'donkeykong',
      'doubledunk',
      'earthworld',
      'elevatoraction',
      'enduro',
      'entombed',
      'et',
      'fishingderby',
      'flagcapture',
      'freeway',
      'frogger',
      'frostbite',
      'galaxian',
      'gopher',
      'gravitar'
      'hangman',
      'hauntedhouse',
      'hero',
      'humancannonball',
      'icehockey',
      'jamesbond',
      'journeyescape',
      'kaboom',
      'kangaroo',
      'keysstonekapers',
      'kingkong',
      'klax',
      'koolaid',
      'krull',
      'kungfumaster',
      'lasergates',
      'lostluggage',
      'mariobros',
      'miniaturegolf',
      'montezumarevenge',
      'mrdo'
      'mspacman',
      'namethisgame',
      'othello',
      'pacman',
      'phoenix',
      'pitfall',
      'pitfall2',
      'pong',
      'pooyan',
      'privateeye',
      'qbert',
      'riverraid',
      'roadrunner'
      'robotank',
      'seaquest',
      'sirlancelot',
      'skiing'
      'solaris',
      'spaceinvaders',
      'spacewar'
      'stargunner',
      'superman',
      'surround',
      'tennis',
      'tetris',
      'tictactoe3d',
      'timepilot',
      'trondead',
      'turmoil',
      'tutankham',
      'upndown',
      'venture',
      'videocheckers',
      'videochess',
      'videocube',
      'videopinball',
      'wizardofwor',
      'wordzapper',
      'yarsrevenge',
      'zaxxon'
    ]
    
    ALL = CLASSIC_CONTROL + MUJOCO + ATARI
    
    env_id = env_id.lower()
    if '-' in env_id:
        env_id = env_id.split('-')[0]
    if 'noframeskip' in env_id:
        env_id = env_id.replace('noframeskip','')
    if '/' in env_id:
        env_id = env_id.split('/')[1]
    
    
    # Check if in one of the three categories
    if env_id not in ALL:
        return None
    
    if env_id in CLASSIC_CONTROL:
        return 'discrete'
    elif env_id in MUJOCO:
        return 'mujoco'
    elif env_id in ATARI:
        return 'atari'
    
    # Shouldnt execute beyond this:
    assert False, "Something is wrong"
    return None
    
def format_atari(env):
    if '/' in env:
        return env.split('/')[1]
    return env
      
      
