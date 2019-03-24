var referencesDataTableColumns = {
    pageReferenceId: "ID",
    title: "Title",
    section: "Relevant Section(s)",
    topic: "Topic",
    subtopic: "Subtopic(s) covered",
    note: "Note(s)",
    url: "Original URL",
    cached: "Cached URL",
};

var references = [
    {
        title: "Introduction to Tensor Calculus, Taha Sochi",
        url: "https://arxiv.org/pdf/1603.01660.pdf",
        cached: "../assets/resources/arXiv/Introduction to Tensor Calculus, Taha Sochi - May 25, 2016 - arXiv 1603.01660v3.pdf",
        contents: [
            {
                section: "1. Notation, Nomenclature and Conventions, bullet point 4",
                topic: "Tensors",
                subtopic: "Tensor Nomenclature (scalars, vectors, dyads, triads, polyads)"
            },
            {
                section: "2.3 Examples of Tensors of Different Ranks",
                topic: "Tensors",
                subtopic: "Examples of tensors of different orders"
            }
        ]
    },
    {
        title: "Kolda, T. G., & Bader, B. W. (2009). Tensor Decompositions and Applications. SIAM Review, 51(3), 455-500. doi:10.1137/07070111x",
        url: "https://epubs.siam.org/doi/abs/10.1137/07070111X?journalCode=siread",
        cached: "../assets/resources/SIAM/Kolda, T. G., & Bader, B. W. (2009). Tensor Decompositions and Applications. SIAM Review, 51(3), 455–500 kolda2009.pdf",
        contents: [
            {
                section: "2. Notation and Preliminaries",
                topic: "Tensors",
                subtopic: ""
            },
            {
                section: "3.1. Tensor Rank",
                topic: "Tensors",
                subtopic: "Rank of a Tensor"
            }
        ]
    }
];