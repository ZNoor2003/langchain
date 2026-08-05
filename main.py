from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()



def main():
    print("Hello from langchain-course!")
    information = """
    Spider-Man: Brand New Day is a 2026 American superhero film based on the Marvel Comics character Spider-Man. Produced by Columbia Pictures, Marvel Studios, and Pascal Pictures, and distributed by Sony Pictures Releasing, it is the 38th film in the Marvel Cinematic Universe (MCU) and the fourth film in the MCU Spider-Man film series following Spider-Man: No Way Home (2021). The film was directed by Destin Daniel Cretton, and written by Chris McKenna and Erik Sommers. Tom Holland stars as Peter Parker / Spider-Man, alongside Zendaya, Sadie Sink, Jacob Batalon, Jon Bernthal, Florence Pugh, Tramell Tillman, Marisa Tomei, and Mark Ruffalo. In the film, Peter anonymously protects New York City as the hero Spider-Man and investigates a powerful new threat while his superpowers undergo a surprising and potentially dangerous evolution.

    Sony was developing a fourth MCU Spider-Man film by August 2019, and producer Amy Pascal revealed in November 2021 that it was intended to be the first in a new trilogy starring Holland. Work on the story began that December, with Holland more involved in development than he was for the previous films. McKenna and Sommers returned as writers by February 2023, Cretton was hired to direct by October 2024, and the title—derived from the 2008 comic book storyline "Brand New Day" that reset Spider-Man's status quo—was announced in March 2025. The producers wanted Holland's Peter to be a "proper Spider-Man" for the first time, on his own and fighting street-level crime in New York City. Ahead of filming, Bernthal and Ruffalo were revealed to be reprising their MCU roles of Frank Castle / Punisher and Bruce Banner / Hulk. Filming took place from August to December 2025, with location filming in Glasgow, Scotland, and throughout England. Soundstage work occurred at Pinewood Studios in England.

    Spider-Man: Brand New Day premiered at the Dolby Theatre in Hollywood, Los Angeles, on July 27, 2026, and was released in the United States on July 31, as part of Phase Six of the MCU. The film received generally positive reviews from critics and has grossed $1.053 billion worldwide, becoming the third-highest-grossing Spider-Man film and the second-highest-grossing film of 2026. It also had the highest opening weekend gross of all time in the United States and Canada and the second-highest worldwide.
    """

    summary_template = """
    Given the information {information}, about the movie Spider-Man: Brand New Day create:
    1. A short summary
    2. Two interesting facts 
    """
    summary_prompt = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
    chain = summary_prompt | llm
    response = chain.invoke({"information": information},config={"callbacks": [langfuse_handler]})
    print(response.content)

if __name__ == "__main__":
    main()
