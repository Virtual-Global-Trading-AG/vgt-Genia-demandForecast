from datetime import datetime, timedelta

class publicHoliday:
    def __init__(self, generator_function:callable, halfday:bool, name:str):
        self.generator_function = generator_function
        self.halfday = halfday
        self.name = name

    def get_date(self, year:int):
        return self.generator_function(year=year)
    
    def __str__(self):
        return self.name

def generator_function_new_year(year:int):
    return datetime(year=year, month=1, day=1)

def generator_function_easter_sunday(year:int):
    """Computus algorithm for calculating easter."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year=year, month=month, day=day)

def generator_function_good_friday(year:int):
    easter = generator_function_easter_sunday.get_date(year=year)
    return easter - timedelta(days=2)

def generator_function_easter_monday(year:int):
    easter = generator_function_easter_sunday(year=year)
    return easter + timedelta(days=1)

def generator_function_ascension_day(year:int):
    easter = generator_function_easter_sunday(year=year)
    return easter + timedelta(days=39)

def generator_function_whit_monday(year:int):
    easter = generator_function_easter_sunday(year=year)
    return easter + timedelta(days=50)

def generator_function_corpus_cristi(year:int):
    easter = generator_function_easter_sunday(year=year)
    return easter + timedelta(days=59)

def generator_function_labor_day(year:int):
    return datetime(year=year, month=5, day=1)

def generator_function_swiss_national_holiday(year:int):
    return datetime(year=year, month=8, day=1)

def generator_function_assumption_of_mary(year:int):
    return datetime(year=year, month=8, day=15)

def generator_function_all_saints(year:int):
    return datetime(year=year, month=11, day=1)

def generator_function_christmas(year:int):
    return datetime(year=year, month=12, day=25)

def generator_function_stephans_day(year:int):
    return datetime(year=year, month=12, day=26)

def generator_function_berchtolds_day(year:int):
    return datetime(year=year, month=1, day=2)


cantons = {
    'SO': [
        publicHoliday(
            generator_function=generator_function_new_year,
            halfday=False,
            name="New Year's"            
        ),
        publicHoliday(
            generator_function=generator_function_good_friday,
            halfday=False,
            name="Good Friday"
        ),
        publicHoliday(
            generator_function=generator_function_easter_sunday,
            halfday=False,
            name="Easter"
        ),
        publicHoliday(
            generator_function=generator_function_easter_monday,
            halfday=False,
            name="Easter monday"
        ),
        publicHoliday(
            generator_function=generator_function_labor_day,
            halfday=False,
            name="Labour Day"
        ),
        publicHoliday(
            generator_function=generator_function_ascension_day,
            halfday=False,
            name="Ascension Day"
        ),
        publicHoliday(
            generator_function=generator_function_whit_monday,
            halfday=False,
            name="Whit Monday"
        ),
        publicHoliday(
            generator_function=generator_function_corpus_cristi,
            halfday=True,
            name="Corpus Cristi"
        ),
        publicHoliday(
            generator_function=generator_function_swiss_national_holiday,
            halfday=False,
            name="National Holiday"
        ),
        publicHoliday(
            generator_function=generator_function_assumption_of_mary,
            halfday=False,
            name="Assumption of Mary"
        ),
        publicHoliday(
            generator_function=generator_function_all_saints,
            halfday=False,
            name="All saints"
        ),
        publicHoliday(
            generator_function=generator_function_christmas,
            halfday=False,
            name="Christmans"
        ),
        publicHoliday(
            generator_function=generator_function_stephans_day,
            halfday=False,
            name="Stephan's Day"
        )
    ],

    'BE': [
        publicHoliday(
            generator_function=generator_function_new_year,
            halfday=False,
            name="New Year's"
        ),
        publicHoliday(
            generator_function=generator_function_berchtolds_day,
            halfday=False,
            name="Berchtold's Day"
        ),
        publicHoliday(
            generator_function=generator_function_good_friday,
            halfday=False,
            name="Good Friday"
        ),
        publicHoliday(
            generator_function=generator_function_easter_sunday,
            halfday=False,
            name="Easter Sunday"
        ),
        publicHoliday(
            generator_function=generator_function_easter_monday,
            halfday=False,
            name="Easter Monday",
        ),
        publicHoliday(
            generator_function=generator_function_ascension_day,
            halfday=False,
            name="Ascension Day"
        ),
        publicHoliday(
            generator_function=generator_function_whit_monday,
            halfday=False,
            name="Whit Monday"
        ),
        publicHoliday(
            generator_function=generator_function_swiss_national_holiday,
            halfday=False,
            name="National Holiday"
        ),
        publicHoliday(
            generator_function=generator_function_christmas,
            halfday=False,
            name="Christmas"
        ),
        publicHoliday(
            generator_function=generator_function_stephans_day,
            halfday=False,
            name="Stephan's Day"
        )
    ],
    'AG': [
        publicHoliday(
            generator_function=generator_function_new_year,
            halfday=False,
            name="New Year's"
        ),
        publicHoliday(
            generator_function=generator_function_berchtolds_day,
            halfday=False,
            name="Berchtold's Day"
        ),
        publicHoliday(
            generator_function=generator_function_good_friday,
            halfday=False,
            name="Good Friday"
        ),
        publicHoliday(
            generator_function=generator_function_easter_sunday,
            halfday=False,
            name="Easter Sunday",
        ),
        publicHoliday(
            generator_function=generator_function_easter_monday,
            halfday=False,
            name="Easter Monday"
        ),
        publicHoliday(
            generator_function=generator_function_labor_day,
            halfday=False,
            name="Labour day"
        ),
        publicHoliday(
            generator_function=generator_function_ascension_day,
            halfday=False,
            name="Ascension Day"
        ),
        publicHoliday(
            generator_function=...
        )
    ]
}   